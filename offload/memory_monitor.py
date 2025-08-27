import torch
import logging
from collections import defaultdict
import sys

logger = logging.getLogger(__name__)

def bytes2human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"

def param_bytes(model):
    m = model.module if hasattr(model, "module") else model
    return sum(p.numel() * p.element_size() for p in m.parameters())

def log_gpu_memory(tag, rank=None):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    log_str = f"GPU Memory {tag}: "
    log_str += f"Allocated: {bytes2human(allocated)} (peak: {bytes2human(peak_allocated)}) | "
    log_str += f"Reserved: {bytes2human(reserved)} (peak: {bytes2human(peak_reserved)})"
    if rank is not None:
        logger.info(f"Rank {rank}: {log_str}")
    else:
        logger.info(log_str)

class MemoryTracker:
    def __init__(self, tag, rank=None):
        self.tag = tag
        self.rank = rank
        self.start_allocated = 0
        self.start_reserved = 0
        self.peak_allocated_before = 0
        self.peak_reserved_before = 0

    def __enter__(self):
        if not torch.cuda.is_available():
            return self
        torch.cuda.synchronize()
        self.start_allocated = torch.cuda.memory_allocated()
        self.start_reserved = torch.cuda.memory_reserved()
        self.peak_allocated_before = torch.cuda.max_memory_allocated()
        self.peak_reserved_before = torch.cuda.max_memory_reserved()
        log_gpu_memory(f"{self.tag}_start", self.rank)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        end_allocated = torch.cuda.memory_allocated()
        end_reserved = torch.cuda.memory_reserved()
        peak_allocated_after = torch.cuda.max_memory_allocated()
        peak_reserved_after = torch.cuda.max_memory_reserved()

        log_str = f"--- Memory Report ({self.tag}) ---"
        log_str += f"\n  Start Allocated: {bytes2human(self.start_allocated)}"
        log_str += f"\n  End Allocated  : {bytes2human(end_allocated)}"
        log_str += f"\n  Peak Allocated : {bytes2human(peak_allocated_after)}"
        log_str += f"\n  Start Reserved : {bytes2human(self.start_reserved)}"
        log_str += f"\n  End Reserved   : {bytes2human(end_reserved)}"
        log_str += f"\n  Peak Reserved  : {bytes2human(peak_reserved_after)}"
        if self.rank is not None:
            logger.info(f"Rank {self.rank}: {log_str}")
        else:
            logger.info(log_str)

def step_memory_report(model, loss, tag, rank=None):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    try:
        if loss is not None and loss.requires_grad:
            loss.backward(retain_graph=True)
    except RuntimeError as e:
        logger.warning(f"Rank {rank}: Could not perform dummy backward for memory report: {e}")
        logger.warning("This might happen if loss does not require grad or graph is already freed.")

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    p_bytes = param_bytes(model)
    act_est = max(peak_alloc - p_bytes, 0)

    log_str = f"=== GPU Memory Report ({tag}) ==="
    log_str += f"\nPeak allocated: {bytes2human(peak_alloc)}"
    log_str += f"\nPeak reserved : {bytes2human(peak_reserved)}"
    log_str += f"\nParams        : {bytes2human(p_bytes)}"
    log_str += f"\n~Activations  : {bytes2human(act_est)}  (rough)"
    if rank is not None:
        logger.info(f"Rank {rank}: {log_str}")
    else:
        logger.info(log_str)

class ActivationMonitor:
    def __init__(self, model, rank=None):
        self.model = model
        self.rank = rank
        self.handles = []
        self.stats = defaultdict(lambda: {"calls": 0, "bytes": 0})
        self.attach_hooks()

    def tensor_bytes(self, t):
        if isinstance(t, (tuple, list)):
            return sum(self.tensor_bytes(x) for x in t)
        if isinstance(t, dict):
            return sum(self.tensor_bytes(v) for v in t.values())
        if torch.is_tensor(t):
            return t.numel() * t.element_size()
        return 0

    def hook_fn(self, name):
        def fn(_mod, _inp, out):
            self.stats[name]["calls"] += 1
            self.stats[name]["bytes"] += self.tensor_bytes(out)
        return fn

    def attach_hooks(self):
        for name, mod in self.model.named_modules():
            if len(list(mod.children())) == 0:
                h = mod.register_forward_hook(self.hook_fn(name))
                self.handles.append(h)

    def report(self, top_k=20):
        if self.rank is not None:
            logger.info(f"Rank {self.rank}: --- Activation Report ---")
        else:
            logger.info("--- Activation Report ---")

        rows = sorted(((k, v["bytes"]) for k, v in self.stats.items()), key=lambda x: x[1], reverse=True)[:top_k]
        for name, b in rows:
            log_str = f"{bytes2human(b):>10}  {name}"
            if self.rank is not None:
                logger.info(f"Rank {self.rank}: {log_str}")
            else:
                logger.info(log_str)

    def cleanup(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.stats.clear()
