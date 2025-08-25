# 内存调试专用配置 - 基于原training_config.py，针对显存分析优化
# Dataset settings
num_frames = None
micro_frame_size = 1  # 保持最小
bbox_mode = 'all-xyz'

# 只用最小分辨率，减少数据加载内存
data_cfg_names = [
    ((224, 400), "Nuscenes_map_cache_box_t_with_n2t_12Hz"),
]

video_lengths_fps = {  # 最小配置
    "224x400": [
        [1], 
        [[120,]],
        [1],  # 不重复
    ]
}

balance_keywords = ["night", "rain", "none"]
dataset_cfg_overrides = [
    (
        # key, value
        ("dataset.dataset_process_root", "./data/nuscenes_mmdet3d-12Hz/"),
        ("dataset.data.train.ann_file", "./data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl"),
        ("dataset.data.val.ann_file", "./data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl"),
        ("dataset.data.train.type", "NuScenesVariableDataset"),
        ("dataset.data.val.type", "NuScenesVariableDataset"),
        ("dataset.data.train.video_length", video_lengths_fps["224x400"][0]),
        ("dataset.data.train.fps", video_lengths_fps["224x400"][1]),
        ("+dataset.data.train.micro_frame_size", micro_frame_size), 
        ("+dataset.data.train.repeat_times", video_lengths_fps["224x400"][2]), 
        ("+dataset.data.train.balance_keywords", balance_keywords), 
        ("dataset.data.val.video_length", video_lengths_fps["224x400"][0]),
        ("dataset.data.val.fps", video_lengths_fps["224x400"][1]),
        ("+dataset.data.val.micro_frame_size", micro_frame_size), 
        ("+dataset.data.val.repeat_times", video_lengths_fps["224x400"][2]), 
        ("+dataset.data.val.balance_keywords", balance_keywords),
    ),
]

# Inference path
bbox2d_path = None

# exp
num_checkpoint_split = 32
tag = ""
outputs = "memory_debug_outputs"  # 专用输出目录
wandb = False

# model
seed = 42
plugin = "zero2-seq"
sp_size = 1  # 不用序列并行，简化内存分析
dtype = "bf16"

# training  
batch_size = 1  # 最小batch size，每个GPU处理1个样本
num_workers = 4  # 适当的worker数量

lr = 1e-4  # 基础学习率，无关紧要，调试用
weight_decay = 0
warmup_steps = 10  # 很少的warmup，快速到达稳定状态
adam_eps = 1e-8

grad_clip = 1.0
grad_checkpoint = False  # 先不用gradient checkpoint，观察原始内存使用

# logging
log_every = 1  # 每步记录
ckpt_every = 0  # 不保存检查点，节省时间和空间

# ColossalAI arguments
micro_batch_size = 1
plugin_extra_flags = dict(
    enable_gradient_accumulation=True,
    empty_init=False,
    precision="bf16",
    stage=2,
    placement_policy='auto',
    cpu_offload=False,  # 先不用CPU offload，观察GPU显存使用
    #initial_scale=2**16,
    #min_scale=1,
    #growth_factor=2,
    #backoff_factor=0.5,
    #growth_interval=1000,
    #enabled=True,
)

bucket_config = {
    "async": True,
}

# zero3
use_zero3 = False

# training
epochs = 1  # 只训练1个epoch
max_train_steps = 5  # 最多5步，重点是内存分析

# additional settings
global_flash_attn = True
global_layernorm = True
global_xformers = True

vae_out_channels = 16

model = dict(
    type="MagicDriveSTDiT3-XL/2",
    simulate_sp_size=[4, 8],
    qk_norm=True,
    pred_sigma=False,
    enable_flash_attn=True and global_flash_attn,
    enable_layernorm_kernel=True and global_layernorm,
    enable_sequence_parallelism=sp_size > 1,
    freeze_y_embedder=True,
    # magicdrive
    with_temp_block=True,
    use_x_control_embedder=True,
    
    # text
    caption_channels=4096,
    model_max_length=300,
    
    # vae
    in_channels=vae_out_channels,
    
    # bbox
    bbox_embedder_param=dict(
        type="ContinuousBBoxWithTextEmbedding",
        latent_size=[1, 28, 50],
        num_cams=6,
        num_frms=16,
        coord_type="lidar",
        use_3d_self_attn=True,
        max_boxes_per_frame=80,
        use_discrete_sin_cos=True,
        freq_reso=10,
        freeze_3d_embedder=False,
        include_text_cond=True,
        include_v_embed=True,
        text_dim=4096,
        bbox_normalizer=[180.0, 55.0, 55.0, 55.0, 180.0, 10.0, 10.0],
        padding_boxes=True,
        enable_xformers=True and global_xformers,
        use_class_id=True,
        num_classes=26,
        max_dim=512,
        bbox_mode=bbox_mode,
        random_sample=False,
        use_bbox_at_first_frame=True,
        enable_bbox_reweighting=False,
        enable_sample_id=True,
    ),
    
    # map
    map_encoder_param=dict(
        type="FramesConv",
        in_channels=7,
        out_channels=8,
        num_groups=8,
        resolution=[180, 320],
        kernels=[7, 7, 7],
        strides=[2, 2, 2],
    ),
    
    # cam
    camera_embedder_param=dict(
        type="CamEmbedderTemp",
        input_dim=3,
        num=4,
    ),
    
    # frame
    frame_emb_param=dict(
        type="FrameEmbedder",
        input_dim=4,
        num=4,
        enable_xformers=True and global_xformers,
    ),
)

# VAE
vae = dict(
    type="CogVideoXVAE3D",
    from_pretrained="./pretrained/CogVideoX-2b",
    micro_batch_size=1,
)

text_encoder = dict(
    type="t5",
    from_pretrained="./pretrained/t5-v1_1-xxl",
    model_max_length=300,
    shardformer=True,
)

scheduler = dict(
    type="ucgm",
    transport_type="ReLinear",
    consistc_ratio=0.0,
    enhanced_target_config=dict(
        enhanced_use_ema=True,
        enhanced_ratio=0.5,
        enhanced_style="fc-vs-fe",
        enhanced_gamma_in=dict(
            type="Linear",
            gamma_min=5e-4,
            gamma_max=1e-2,
        ),
        enhanced_alpha_in=dict(
            type="Linear",
            alpha_min=0.0001,
            alpha_max=3.0,
        ),
    ),
    gamma_in=dict(
        type="PowerCosine",
        power=1.0,
        gamma_min=5e-4,
        gamma_max=1e-2,
    ),
    alpha_in=dict(
        type="PowerCosine",
        power=1.0,
        alpha_min=0.0001,
        alpha_max=3.0,
    ),
    loss_weighting=dict(
        type="P2LossWeight",
        gamma_min=5e-4,
        gamma_max=1e-2,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=0.0,
        s_noise=1.0,
    ),
)

# validation
val = dict(
    enabled=False,  # 禁用验证，专注内存分析
    every=10000,
    num_workers=1,
    batch_size=1,
    samples=dict(
        seed=42,
        num_sample=1,
        sample_H=224,
        sample_W=400,
        sample_T=1,
        save_fps=8,
    ),
    validation_index=list(range(8)),
    scheduler=dict(
        type="dpm-solver",
        num_sampling_steps=20,
        cfg_scale=6.0,
    ),
    verbose=1,
)

# load
load = None
partial_load = None

# debug settings
debug = False  # 使用内存调试模式代替
debug_memory = True  # 启用详细内存分析
debug_activation = False  # 激活值监控（可能很慢）
verbose_mode = False

# early termination
report_every = 0  # 不做验证报告
drop_cond_ratio = 0.1
drop_cond_ratio_t = 0.1

# Mixed Precision
loss_scale_config = dict(
    initial_scale=65536,
    min_scale=1,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=100,
    enabled=False,
)

# EMA
ema_decay = 0.999

# Memory debugging specific
memory_monitoring = dict(
    enable_detailed=True,
    report_every_step=True,
    enable_activation_analysis=False,  # 太慢，先不用
    max_debug_steps=5,  # 最多调试5步
)
