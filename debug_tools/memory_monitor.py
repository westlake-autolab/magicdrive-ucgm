"""
GPU内存和激活值监控工具
用于MagicDrive训练的显存分析和优化
"""

import torch
from collections import defaultdict
import logging

def bytes2human(n):
    """将字节数转换为人类可读格式"""
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024: 
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"

def param_bytes(model):
    """计算模型参数占用的内存"""
    m = model.module if hasattr(model, "module") else model
    return sum(p.numel() * p.element_size() for p in m.parameters())

def step_memory_report(model, step_name="step"):
    """
    A) 粗粒度：整步峰值显存监控
    最实用的方法，快速定位是否超显存
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # 记录当前状态
    current_alloc = torch.cuda.memory_allocated()
    current_reserved = torch.cuda.memory_reserved()
    
    def get_peak_stats():
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        
        p_bytes = param_bytes(model)
        act_est = max(peak_alloc - p_bytes, 0)
        
        print(f"=== GPU Memory Report ({step_name}) ===")
        print(f"Peak allocated: {bytes2human(peak_alloc)}")
        print(f"Peak reserved : {bytes2human(peak_reserved)}")
        print(f"Params        : {bytes2human(p_bytes)}")
        print(f"~Activations  : {bytes2human(act_est)}  (rough)")
        print(f"Current alloc : {bytes2human(current_alloc)}")
        print("=" * 50)
        
        return {
            'peak_alloc': peak_alloc,
            'peak_reserved': peak_reserved,
            'params': p_bytes,
            'activations_est': act_est
        }
    
    return get_peak_stats

def tensor_bytes(t):
    """计算张量或张量集合的字节数"""
    if isinstance(t, (tuple, list)):
        return sum(tensor_bytes(x) for x in t)
    if isinstance(t, dict):
        return sum(tensor_bytes(v) for v in t.values())
    if torch.is_tensor(t):
        return t.numel() * t.element_size()
    return 0

class ActivationMonitor:
    """
    B) 分层观测：forward hooks 看每层输出
    用于定位哪一层最吃激活显存
    """
    def __init__(self, model, monitor_attention=True):
        self.handles = []
        self.stats = defaultdict(lambda: {"calls": 0, "bytes": 0})
        self.monitor_attention = monitor_attention
        self.attach_hooks(model)
    
    def attach_hooks(self, model):
        def hook(name):
            def fn(_mod, _inp, out):
                self.stats[name]["calls"] += 1
                self.stats[name]["bytes"] += tensor_bytes(out)
            return fn
        
        for name, mod in model.named_modules():
            # 重点监控注意力层、MLP层、归一化层
            if (len(list(mod.children())) == 0 and 
                (any(key in name.lower() for key in ['attn', 'mlp', 'norm', 'linear', 'conv']) or
                 self.monitor_attention)):
                h = mod.register_forward_hook(hook(name))
                self.handles.append(h)
    
    def get_top_layers(self, top_k=20):
        """获取显存占用最大的前K层"""
        rows = sorted(
            ((k, v["bytes"]) for k, v in self.stats.items()), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        return rows
    
    def print_report(self, top_k=20):
        """打印分层显存报告"""
        print(f"=== Top {top_k} Layers by Activation Size ===")
        rows = self.get_top_layers(top_k)
        for name, b in rows:
            print(f"{bytes2human(b):>10}  {name}")
        print("=" * 60)
    
    def cleanup(self):
        """清理hooks"""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.stats.clear()

class MemoryTracker:
    """简化的内存追踪器，用于训练过程中的关键点监控"""
    
    def __init__(self, model):
        self.model = model
        self.checkpoints = {}
    
    def checkpoint(self, name):
        """在关键点记录内存使用"""
        torch.cuda.synchronize()
        self.checkpoints[name] = {
            'allocated': torch.cuda.memory_allocated(),
            'reserved': torch.cuda.memory_reserved(),
            'max_allocated': torch.cuda.max_memory_allocated()
        }
    
    def print_diff(self, name1, name2):
        """打印两个检查点之间的内存差异"""
        if name1 not in self.checkpoints or name2 not in self.checkpoints:
            print(f"Checkpoint {name1} or {name2} not found")
            return
        
        cp1 = self.checkpoints[name1]
        cp2 = self.checkpoints[name2]
        
        alloc_diff = cp2['allocated'] - cp1['allocated']
        reserved_diff = cp2['reserved'] - cp1['reserved']
        
        print(f"=== Memory diff: {name1} -> {name2} ===")
        print(f"Allocated diff: {bytes2human(alloc_diff)}")
        print(f"Reserved diff : {bytes2human(reserved_diff)}")
        print("=" * 40)

# 快速使用的便捷函数
def quick_memory_check():
    """快速检查当前GPU内存状态"""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    
    print(f"Current GPU Memory:")
    print(f"  Allocated: {bytes2human(allocated)}")
    print(f"  Reserved:  {bytes2human(reserved)}")
    print(f"  Peak:      {bytes2human(max_allocated)}")

# 用于在训练中集成的简化版本
def log_gpu_memory(step_name="", rank=None):
    """简化版本，适合在训练循环中调用"""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    max_allocated = torch.cuda.max_memory_allocated()
    
    rank_info = f"[Rank {rank}] " if rank is not None else ""
    logging.info(f"{rank_info}GPU Memory {step_name}: {bytes2human(allocated)} (peak: {bytes2human(max_allocated)})")






