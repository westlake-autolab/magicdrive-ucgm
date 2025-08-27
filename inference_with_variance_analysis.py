#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带方差分析的MagicDrive-V2推理脚本

这是一个修改版的推理脚本，集成了x参数方差分析功能。
基于原始的 scripts/inference_magicdrive.py，添加了方差分析补丁。

使用方法：
torchrun --standalone --nproc_per_node ${GPUS} inference_with_variance_analysis.py ${CFG} \
    --cfg-options model.from_pretrained=${PATH_TO_MODEL} num_frames=${FRAME} \
    cpu_offload=true scheduler.type=rflow-slice
"""

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

import colossalai
import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 导入MagicDrive相关模块
from magicdrivedit.acceleration.parallel_states import (
    get_data_parallel_group,
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    is_distributed,
    is_main_process,
)
from magicdrivedit.datasets import DATASETS
from magicdrivedit.models import MODELS
from magicdrivedit.registry import build_module
from magicdrivedit.schedulers import SCHEDULERS
from magicdrivedit.utils.config_utils import parse_configs
from magicdrivedit.utils.misc import (
    all_reduce_mean,
    create_logger,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
)

# 导入方差分析补丁
from variance_analysis_patch import apply_variance_analysis_patch


def main():
    # =============================
    # 1. 配置解析和环境设置
    # =============================
    cfg = parse_configs(training=False)
    print("Configuration loaded successfully!")
    
    # 设置随机种子
    set_random_seed(seed=cfg.get("seed", 1024))
    
    # 设备和数据类型配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.get("dtype", "fp16"))
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # =============================
    # 2. 分布式环境初始化
    # =============================
    if is_distributed():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        colossalai.launch_from_torch({})
    
    # =============================
    # 3. 数据集和数据加载器构建
    # =============================
    print("Building dataset...")
    
    # 数据集配置
    dataset_cfg = cfg.dataset
    if hasattr(dataset_cfg, "img_collate_param"):
        dataset_cfg.img_collate_param.is_train = False
    
    # 构建数据集
    dataset = build_module(dataset_cfg, DATASETS)
    
    # 验证索引处理
    val_indices = cfg.get("val_indices", "all")
    if val_indices == "even":
        dataset.samples = [dataset.samples[i] for i in range(0, len(dataset.samples), 2)]
    elif val_indices == "odd":
        dataset.samples = [dataset.samples[i] for i in range(1, len(dataset.samples), 2)]
    
    print(f"Dataset size: {len(dataset)}")
    
    # 构建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 推理时通常使用batch_size=1
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )
    
    # =============================
    # 4. 模型构建
    # =============================
    print("Building model...")
    
    # 文本编码器
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device, dtype=dtype)
    
    # VAE模型
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    
    # 主要的扩散模型
    input_size = (cfg.num_frames, *cfg.image_size)
    latent_size = vae.get_latent_size(input_size)
    
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            dtype=dtype,
        )
        .to(device, dtype)
        .eval()
    )
    
    print(f"Model parameters: {format_numel_str(get_model_numel(model))}")
    
    # =============================
    # 5. 应用方差分析补丁 🔥
    # =============================
    print("\n" + "="*60)
    print("应用x参数方差分析补丁...")
    print("="*60)
    
    # 创建方差分析日志目录
    variance_log_dir = cfg.get("variance_log_dir", "./variance_logs")
    os.makedirs(variance_log_dir, exist_ok=True)
    
    # 应用方差分析补丁
    model = apply_variance_analysis_patch(
        model, 
        save_log=True, 
        log_dir=variance_log_dir
    )
    
    print("方差分析补丁应用完成！")
    print("="*60 + "\n")
    
    # =============================
    # 6. 调度器构建
    # =============================
    print("Building scheduler...")
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    # =============================
    # 7. CPU卸载配置
    # =============================
    if cfg.get("cpu_offload", False):
        print("Enabling CPU offload...")
        # 这里可以添加CPU卸载的具体实现
    
    # =============================
    # 8. 推理循环
    # =============================
    print("Starting inference...")
    
    # 创建输出目录
    output_dir = cfg.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 推理参数
    num_frames = cfg.get("num_frames", 20)
    height, width = cfg.image_size
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Inference")):
            print(f"\n{'='*50}")
            print(f"Processing batch {i+1}/{len(dataloader)}")
            print(f"{'='*50}")
            
            # 数据预处理
            y = batch.pop("captions")[0] if "captions" in batch else [""]
            maps = batch.pop("bev_map_with_aux").to(device, dtype) if "bev_map_with_aux" in batch else None
            bbox = batch.pop("bboxes_3d_data") if "bboxes_3d_data" in batch else None
            cams = batch.pop("cams") if "cams" in batch else None
            rel_pos = batch.pop("rel_pos") if "rel_pos" in batch else None
            
            # 生成随机噪声作为初始潜在表示
            # 这里的z就是会传递给模型的x参数
            z_shape = (1, vae.out_channels * 6, num_frames, height // 8, width // 8)  # 6个视角
            z = torch.randn(z_shape, device=device, dtype=dtype)
            
            print(f"Initial latent z shape: {z.shape}")
            print(f"Initial latent z variance: {torch.var(z).item():.6f}")
            
            # 准备模型参数
            model_args = {
                "maps": maps,
                "bbox": bbox,
                "cams": cams,
                "rel_pos": rel_pos,
                "fps": cfg.get("fps", 8),
                "height": height,
                "width": width,
                "num_frames": num_frames,
            }
            
            # 移除None值
            model_args = {k: v for k, v in model_args.items() if v is not None}
            
            print(f"Model args keys: {list(model_args.keys())}")
            
            # 调度器采样
            print("\nStarting scheduler sampling...")
            print("-" * 30)
            
            try:
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=y,
                    device=device,
                    additional_args=model_args,
                )
                
                print(f"\nSampling completed! Output shape: {samples.shape}")
                print(f"Output variance: {torch.var(samples).item():.6f}")
                
                # VAE解码
                print("Decoding with VAE...")
                decoded_samples = vae.decode(samples)
                
                print(f"Decoded samples shape: {decoded_samples.shape}")
                print(f"Decoded samples range: [{decoded_samples.min().item():.3f}, {decoded_samples.max().item():.3f}]")
                
                # 保存结果（这里可以添加具体的保存逻辑）
                # save_samples(decoded_samples, output_dir, i)
                
            except Exception as e:
                print(f"Error during sampling: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            print(f"Batch {i+1} completed!")
            
            # 限制处理的批次数量（用于测试）
            if cfg.get("max_batches", None) and i >= cfg.max_batches - 1:
                print(f"Reached maximum batches limit: {cfg.max_batches}")
                break
    
    # =============================
    # 9. 保存方差分析日志
    # =============================
    print("\n" + "="*50)
    print("保存方差分析日志...")
    print("="*50)
    
    if hasattr(model, 'save_variance_log'):
        model.save_variance_log()
        print(f"方差分析日志已保存到: {variance_log_dir}")
    
    print("推理完成！")
    print("="*50)


if __name__ == "__main__":
    main()