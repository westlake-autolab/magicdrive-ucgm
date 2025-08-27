#!/usr/bin/env python3#
# -*- coding: utf-8 -*-
"""
MagicDrive-V2  ──  x  参数方差分析补丁  (2025-08-04 rev2)

需求 ✔
1. 仅当
      • step_count % print_every == 0   或
      • diffusion timestep < final_n_steps
   时才打印。
2. JSON 里去掉巨大数组（channel / temporal / camera / sequence / feature variance），
   只保留全局统计，文件大幅减小。

用法（示例）：
    from variance_analysis_patch import apply_variance_analysis_patch
    model = apply_variance_analysis_patch(
                model,
                print_every=100,     # ← 每 100 次统计打印一次
                final_n_steps=20     # ← 每张 sample 最后 20 个扩散 step 强制打印
            )
    ...
    model.save_variance_log()
"""

from __future__ import annotations
import os, json, torch, numpy as np, types
from typing import Dict, List
from datetime import datetime

# ------------------------------------------------------------
# 1)  方差分析器
# ------------------------------------------------------------
class XVarianceAnalyzer:
    def __init__(self,
                 save_log:     bool = True,
                 log_dir:      str  = "./variance_logs",
                 print_every:  int  = 100,
                 final_n_steps:int  = 1500):
        self.save_log   = save_log
        self.log_dir    = log_dir
        self.print_every = print_every
        self.final_n_steps = final_n_steps
        self.step_count = 0
        self.variance_history: List[Dict] = []
        self._force_print_cur_step = False      # 供 hooks 共享

        if save_log and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    # --------  核心统计  --------
    def analyze_tensor_variance(self,
                                tensor: torch.Tensor,
                                name:   str,
                                *,
                                force_print: bool | None = None) -> Dict:
        """
        • 将 tensor 转成 fp32 CPU，避免 BF16/F16 限制并腾出 GPU 显存
        • 仅保存/返回全局统计，**不再**存储庞大数组
        """
        if force_print is None:          # hooks 走这里
            force_print = self._force_print_cur_step

        step = self.step_count
        self.step_count += 1

        t = tensor.detach().to(torch.float32).cpu()

        stats: Dict = dict(
            name   = name,
            step   = step,
            shape  = list(t.shape),
            dtype  = str(tensor.dtype),
            device = str(tensor.device),
            global_variance = float(torch.var(t)),
            global_mean     = float(torch.mean(t)),
            global_std      = float(torch.std(t)),
            min_value       = float(torch.min(t)),
            max_value       = float(torch.max(t)),
        )

        self.variance_history.append(stats)

        # 打印策略
        if force_print or (step % self.print_every == 0):
            self._print_summary(stats)

        return stats

    # --------  打印摘要  --------
    @staticmethod
    def _print_summary(s: Dict):
        print(f"\n=== {s['name']} | step {s['step']} ===")
        print(f"shape {s['shape']}  dtype {s['dtype']}  device {s['device']}")
        print(f"global var {s['global_variance']:.5e}  "
              f"mean {s['global_mean']:.3e}  std {s['global_std']:.3e}  "
              f"range [{s['min_value']:.2e}, {s['max_value']:.2e}]")

    # --------  保存日志  --------
    def save_variance_log(self, filename: str | None = None):
        if not self.save_log:
            return
        if filename is None:
            filename = f"variance_{datetime.now():%Y%m%d_%H%M%S}.json"
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(self.variance_history, f, indent=2)
        print(f"[XVarianceAnalyzer] log saved → {path}")


# ------------------------------------------------------------
# 2)  给 MagicDrive 模型打补丁
# ------------------------------------------------------------
def _make_forward_wrapper(model, analyzer: XVarianceAnalyzer):
    orig_forward = model.forward        # 已绑定 self

    def forward_with_var(self, x, timestep, y, maps, bbox, cams, rel_pos, fps,
                         height, width, drop_cond_mask=None, drop_frame_mask=None,
                         mv_order_map=None, t_order_map=None,
                         mask=None, x_mask=None, **kw):

        # --- 判断是否处于「最后 n 个扩散 step」 ---
        # timestep 可能是 int，也可能是 shape=[B] 的张量
        cur_t = int(timestep[0].item() if torch.is_tensor(timestep) else timestep)
        force = cur_t > analyzer.final_n_steps
        analyzer._force_print_cur_step = force

        # 输入 / 输出
        analyzer.analyze_tensor_variance(x, "input_x", force_print=force)

        out = orig_forward(x, timestep, y, maps, bbox, cams, rel_pos, fps,
                           height, width, drop_cond_mask, drop_frame_mask,
                           mv_order_map, t_order_map, mask, x_mask, **kw)

        if isinstance(out, torch.Tensor):
            analyzer.analyze_tensor_variance(out, "output_x", force_print=force)

        analyzer._force_print_cur_step = False   # 还原
        return out

    return types.MethodType(forward_with_var, model)


def _patch_controlnet_blocks(model, analyzer: XVarianceAnalyzer):
    def make_hook(name):
        def hook(_, __, out):
            # hooks 里不清楚 timestep，交给 analyzer 内部的 _force_print_cur_step
            if isinstance(out, (tuple, list)):
                x, x_skip = out
                analyzer.analyze_tensor_variance(x,      f"{name}_x")
                analyzer.analyze_tensor_variance(x_skip, f"{name}_xskip")
            else:
                analyzer.analyze_tensor_variance(out, name)
        return hook

    for attr in ("control_blocks_s", "control_blocks_t"):
        if hasattr(model, attr):
            for i, blk in enumerate(getattr(model, attr)):
                blk.register_forward_hook(make_hook(f"{attr}_{i}"))

    for name in ("x_embedder", "before_proj"):
        if hasattr(model, name):
            getattr(model, name).register_forward_hook(make_hook(name))


def apply_variance_analysis_patch(model,
                                  *,
                                  save_log:     bool = True,
                                  log_dir:      str  = "./variance_logs",
                                  print_every:  int  = 100,
                                  final_n_steps:int  = 20):
    """
    返回打好补丁的模型。外部可用：
        model._variance_analyzer
        model.save_variance_log()
    """
    print("="*60)
    print("▶  Variance-analysis patch applied")
    print(f"    • print_every   = {print_every}")
    print(f"    • final_n_steps = {final_n_steps}")
    print(f"    • log dir       = {log_dir}")
    print("="*60 + "\n")

    analyzer = XVarianceAnalyzer(save_log, log_dir, print_every, final_n_steps)
    model.forward = _make_forward_wrapper(model, analyzer)
    _patch_controlnet_blocks(model, analyzer)

    model._variance_analyzer = analyzer
    model.save_variance_log  = analyzer.save_variance_log
    return model


# ------------------------------------------------------------
# quick self-test
# ------------------------------------------------------------
if __name__ == "__main__":
    print("▲  本文件仅作为补丁模块使用，请在推理脚本中 import 之后调用。")


# 使用示例
if __name__ == "__main__":
    print("""
    MagicDrive-V2 x参数方差分析补丁使用说明：
    
    1. 在推理脚本中导入此模块：
       from variance_analysis_patch import apply_variance_analysis_patch
    
    2. 在模型构建后应用补丁：
       model = apply_variance_analysis_patch(model)
    
    3. 运行推理，方差信息会自动打印并保存
    
    4. 推理结束后保存完整日志：
       model.save_variance_log()
    """)