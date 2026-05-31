from collections import defaultdict
from contextlib import contextmanager
import torch

class CudaTimer:
    def __init__(self, enabled: bool, device=None):
        self.enabled = enabled and torch.cuda.is_available()
        self.device = device
        self.ms = defaultdict(float)

    @contextmanager
    def timeit(self, key: str):
        if not self.enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        self.ms[key] += start.elapsed_time(end)

    def reset(self):
        self.ms.clear()

    def summary(self, topk: int = 30):
        items = sorted(self.ms.items(), key=lambda x: -x[1])[:topk]
        return items

@torch.no_grad()
def _cuda_ms(fn, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)  # ms

