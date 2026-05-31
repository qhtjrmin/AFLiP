# encoder_cache.py
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from torch_scatter import scatter
from utils.cuda_timer import CUDATimer

import flip_backend as aflip_backend

# --- timing utils ---
from collections import defaultdict

class OpStats:
    """accumulate timings"""
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.ms = defaultdict(float)

    def add(self, key: str, v_ms: float):
        if self.enabled:
            self.ms[key] += float(v_ms)

    def reset(self):
        self.ms.clear()

    def report(self, topk: int = 50):
        if not self.enabled:
            return
        items = sorted(self.ms.items(), key=lambda kv: kv[1], reverse=True)
        print("===== encoder op timing breakdown (ms, accumulated) =====")
        for k, v in items[:topk]:
            print(f"{k:<40} {v:>12.3f} ms")
        print("=========================================================")

@contextmanager
def maybe_cuda_timer(stats, key, device):
    if stats is not None and getattr(stats, "enabled", False):
        t = CUDATimer(True, device)
        with t:
            yield
        stats.add(key, t.ms())
    else:
        yield

# -----------------------------
# CUDA mean+history aggregation wrapper
# -----------------------------

def _aggr_mean_dense(
    x: torch.Tensor,           # [N, d]
    ptr: torch.Tensor,         # [N+1] int64
    idx: torch.Tensor,         # [E]   int64
    num_nodes: int,
    frontier_rows: torch.Tensor,
    prev_rows: torch.Tensor,
    history_pack: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    stats: Optional[OpStats] = None,
    tag: str = "aggr_dense"
) -> torch.Tensor:
    edge_value = torch.empty(0, device=x.device, dtype=x.dtype)
    
    with maybe_cuda_timer(stats, tag, x.device):
        if history_pack is None:
            out = aflip_backend.aggr_forward_frontier_dense(ptr, idx, x, edge_value, frontier_rows, prev_rows)
        else:
            hmap, hbuf = history_pack
            out = aflip_backend.aggr_forward_history_frontier_dense(ptr, idx, x, hmap, hbuf, frontier_rows, edge_value, prev_rows)
    return out

def _aggr_mean_with_history(
    x: torch.Tensor,           # [N, d]
    ptr: torch.Tensor,         # [N+1] int64
    idx: torch.Tensor,         # [E]   int64
    num_nodes: int,
    hmap: torch.Tensor,        # [F]   int64, if -1, active(aggregation), if >=0, cache pull
    hbuf: torch.Tensor,        # [N, d] float
    frontier_rows: torch.Tensor, # [F] int
    stats: Optional[OpStats] = None,
    tag: str = "aggr_history"
) -> torch.Tensor:
    edge_value = torch.empty(0, device=x.device, dtype=x.dtype)
    
    if stats is not None and stats.enabled:
        timer = CUDATimer(stats.enabled, x.device)
        with timer:
            if frontier_rows is None:
                out = aflip_backend.aggr_forward_history(ptr, idx, num_nodes, x, hmap, hbuf, edge_value)
            else:
                out = aflip_backend.aggr_forward_history_frontier(ptr, idx, x, hmap, hbuf, frontier_rows, edge_value)
        stats.add(tag, timer.ms())
    else:
        if frontier_rows is None:
            out = aflip_backend.aggr_forward_history(ptr, idx, num_nodes, x, hmap, hbuf, edge_value)
        else:
            out = aflip_backend.aggr_forward_history_frontier(ptr, idx, x, hmap, hbuf, frontier_rows, edge_value)
    return out

def _aggr_mean(
    x: torch.Tensor,           # [N, d]
    ptr: torch.Tensor,         # [N+1] int64
    idx: torch.Tensor,         # [E]   int64
    num_nodes: int,
    frontier_rows: torch.Tensor,
    stats: Optional[OpStats] = None,
    tag: str = "aggr_plain"
) -> torch.Tensor:
    edge_value = torch.empty(0, device=x.device, dtype=x.dtype)
    
    with maybe_cuda_timer(stats, tag, x.device):
        if frontier_rows is None:
            out = aflip_backend.aggr_forward_cuda(ptr, idx, num_nodes, x, edge_value)
        else:
            out = aflip_backend.aggr_forward_frontier(ptr, idx, x, frontier_rows, edge_value)
    return out

def _aggr_weighted_with_history(
    x: torch.Tensor,           # [N, d]
    ptr: torch.Tensor,         # [N+1] int64
    idx: torch.Tensor,         # [E]   int64
    num_nodes: int,
    hmap: torch.Tensor,        # [N]   int64, if -1, active(aggregation), if >=0, cache pull
    hbuf: torch.Tensor,        # [N, d] float
    frontier_rows: torch.Tensor, # [F] int
    edge_value: torch.Tensor,   # [E]   float
    stats: Optional[OpStats] = None,
    tag: str = "aggr_weighted_history"
) -> torch.Tensor:
    with maybe_cuda_timer(stats, tag, x.device):
        if frontier_rows is None:
            print("NOT IMPLEMETED")
            out = aflip_backend.aggr_forward_history(ptr, idx, num_nodes, x, hmap, hbuf, edge_value)
        else:
            out = aflip_backend.aggr_forward_history_frontier(ptr, idx, x, hmap, hbuf, frontier_rows, edge_value)
    return out

def _aggr_weighted(
    x: torch.Tensor,           # [N, d]
    ptr: torch.Tensor,         # [N+1] int64
    idx: torch.Tensor,         # [E]   int64
    num_nodes: int,
    frontier_rows: torch.Tensor,
    edge_value: torch.Tensor,
    stats: Optional[OpStats] = None,
    tag: str = "aggr_weighted"
) -> torch.Tensor:
    with maybe_cuda_timer(stats, tag, x.device):
        if frontier_rows is None:
            out = aflip_backend.aggr_forward_cuda(ptr, idx, num_nodes, x, edge_value)
        else:
            out = aflip_backend.aggr_forward_frontier(ptr, idx, x, frontier_rows, edge_value)
    return out

def aggr_mean_pyg(x, ptr, idx, num_nodes):
    device = x.device

    row_counts = (ptr[1:] - ptr[:-1]).to(torch.long)

    row = torch.repeat_interleave(
        torch.arange(num_nodes, device=device, dtype=torch.long),
        row_counts
    )  # [E]

    col = idx.to(torch.long)

    out = scatter(x[col], row, dim=0, dim_size=num_nodes, reduce='mean')
    return out

# -----------------------------
# Layer: GraphSAGE mean aggregation + linear (+ root)
# -----------------------------
class HistSageLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        root_weight: bool,
        bias: bool,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.root_weight = bool(root_weight)

        self.lin_l = nn.Linear(self.in_channels, self.out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = nn.Linear(self.in_channels, self.out_channels, bias=False)
        else:
            self.lin_r = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.lin_r is not None:
            self.lin_r.reset_parameters()

    def aggregate_only(
        self,
        x: torch.Tensor,
        ptr: torch.Tensor,
        idx: torch.Tensor,
        num_nodes: int,
        history_pack: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        frontier_rows: Optional[torch.Tensor] = None,
        stats: Optional[OpStats] = None,
        prev_rows: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
    ):
        if prev_rows is None:
            if history_pack is None:
                aggr = _aggr_mean(
                    x, ptr, idx, num_nodes,
                    frontier_rows=frontier_rows,
                    stats=stats,
                    tag=f"layer{layer_idx}.aggr_plain",
                )
            else:
                hmap, hbuf = history_pack
                aggr = _aggr_mean_with_history(
                    x, ptr, idx, num_nodes,
                    hmap, hbuf,
                    frontier_rows,
                    stats=stats,
                    tag=f"layer{layer_idx}.aggr_history",
                )
        else:
            aggr = _aggr_mean_dense(
                x, ptr, idx, num_nodes,
                frontier_rows,
                prev_rows,
                history_pack,
                stats=stats,
                tag=f"layer{layer_idx}.aggr_dense",
            )

        return aggr

    def select_root_x(
        self,
        x: torch.Tensor,
        num_nodes: int,
        frontier_rows: Optional[torch.Tensor],
        prev_rows: Optional[torch.Tensor],
    ):
        if not self.root_weight:
            return None

        if frontier_rows is None:
            return x[:num_nodes]

        fr = frontier_rows.to(torch.long)

        if prev_rows is None:
            return x[fr]

        pr = prev_rows.to(torch.long)
        fr_idx = pr[fr]
        return x[fr_idx]

    def transform_only(
        self,
        aggr: torch.Tensor,
        root_x: Optional[torch.Tensor] = None,
    ):
        out_l = self.lin_l(aggr)
        out = out_l

        if self.root_weight:
            out = out + self.lin_r(root_x)

        return out, out_l

    def forward(
        self,
        x: torch.Tensor,
        ptr: torch.Tensor,
        idx: torch.Tensor,
        num_nodes: int,
        history_pack: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        frontier_rows: Optional[torch.Tensor] = None,
        stats: Optional[OpStats] = None,
        prev_rows: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
    ):
        aggr = self.aggregate_only(
            x=x,
            ptr=ptr,
            idx=idx,
            num_nodes=num_nodes,
            history_pack=history_pack,
            frontier_rows=frontier_rows,
            stats=stats,
            prev_rows=prev_rows,
            layer_idx=layer_idx,
        )

        root_x = self.select_root_x(
            x=x,
            num_nodes=num_nodes,
            frontier_rows=frontier_rows,
            prev_rows=prev_rows,
        )

        out, out_l = self.transform_only(aggr, root_x)

        his = aggr
        return out, his, out_l

# -----------------------------
# Encoder model: (L-layer) Cached SAGE
# -----------------------------
class BaseCacheEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        use_bn: bool = True,
        max_x: int = -1,
    ):
        super().__init__()
        assert num_layers >= 1
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)

        if max_x >= 0:
            tmp = nn.Embedding(max_x + 1, hidden_channels)
            nn.init.orthogonal_(tmp.weight)
            self.xemb = nn.Sequential(tmp, nn.Dropout(dropout))
        else:
            self.xemb = None
            
        self.convs = nn.ModuleList()
        # Subclass must populate self.convs
        
        self.lns = nn.ModuleList()
        if self.use_bn:
            for l in range(self.num_layers):
                dim = out_channels if (l == self.num_layers - 1) else hidden_channels
                self.lns.append(nn.LayerNorm(dim))

        self.opstats = OpStats(enabled=False)

    def enable_timing(self):
        self.opstats.enabled = True

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def scatter_to_full(self, out_f: torch.Tensor, frontier_rows: torch.Tensor, num_nodes: int) -> torch.Tensor:
        full = out_f.new_zeros((num_nodes, out_f.size(1)))
        full[frontier_rows.to(torch.long)] = out_f
        return full

    def _call_conv(self, layer_idx, x, ptr, idx, num_nodes, history_pack, frontier_rows, mapping_rows=None, is_mapping=False):
        raise NotImplementedError

    def _post_aggr_block(
        self,
        aggr: torch.Tensor,
        root_x: Optional[torch.Tensor],
        layer_idx: int,
        is_last: bool,
    ):
        conv = self.convs[layer_idx]

        out, out_l = conv.transform_only(aggr, root_x)

        if self.use_bn:
            out = self.lns[layer_idx](out)

        if not is_last:
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)

        return out, out_l

    def forward_layer(
        self,
        layer_idx: int,
        x: torch.Tensor,
        ptr: torch.Tensor,
        idx: torch.Tensor,
        num_nodes: int,
        history_pack: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        frontier_rows: Optional[torch.Tensor] = None,
        mapping_rows: Optional[torch.Tensor] = None,
        is_mapping: bool = False,
    ):
        stats = self.opstats
        dev = x.device

        if layer_idx == 0 and self.xemb is not None:
            x = self.xemb(x)

        conv = self.convs[layer_idx]
        is_last = layer_idx == self.num_layers - 1

        # 1) Aggregation.
        t = CUDATimer(stats.enabled, dev)
        with t:
            aggr = conv.aggregate_only(
                x=x,
                ptr=ptr,
                idx=idx,
                num_nodes=num_nodes,
                history_pack=history_pack,
                frontier_rows=frontier_rows,
                stats=stats,
                prev_rows=mapping_rows,
                layer_idx=layer_idx,
            )
        stats.add(f"layer{layer_idx}.aggr_only_total", t.ms())

        his = aggr

        # 2) Root feature selection.
        root_x = conv.select_root_x(
            x=x,
            num_nodes=num_nodes,
            frontier_rows=frontier_rows,
            prev_rows=mapping_rows,
        )

        # 3) Linear/root transform + normalization + activation/dropout.
        t = CUDATimer(stats.enabled, dev)
        with t:
            out, extra = self._post_aggr_block(
                aggr,
                root_x,
                layer_idx,
                is_last,
            )
        stats.add(f"layer{layer_idx}.post_aggr_block", t.ms())

        # 4) 마지막 layer 처리
        if is_last:
            if not is_mapping and frontier_rows is not None:
                t = CUDATimer(stats.enabled, dev)
                with t:
                    out = self.scatter_to_full(out, frontier_rows, num_nodes)
                stats.add(f"layer{layer_idx}.scatter_to_full", t.ms())
            return out, his, extra

        # 5) non-last layer scatter
        if not is_mapping and frontier_rows is not None:
            t = CUDATimer(stats.enabled, dev)
            with t:
                out = self.scatter_to_full(out, frontier_rows, num_nodes)
            stats.add(f"layer{layer_idx}.scatter_to_full", t.ms())

        return out, his, extra

    def forward(self, x, ptr, idx, num_nodes):
        for l in range(self.num_layers):
            x, _, _ = self.forward_layer(l, x, ptr, idx, num_nodes, history_pack=None, frontier_rows=None)
        return x

class CacheSAGEEncoder(BaseCacheEncoder):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        root_weight: bool = True,
        bias: bool = True,
        use_bn: bool = True,
        max_x=-1
    ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, use_bn, max_x)
        
        in_c = hidden_channels if max_x >= 0 else in_channels
        
        if self.num_layers == 1:
            self.convs.append(HistSageLayer(in_c, out_channels, root_weight, bias))
        else:
            self.convs.append(HistSageLayer(in_c, hidden_channels, root_weight, bias))
            for _ in range(self.num_layers - 2):
                self.convs.append(HistSageLayer(hidden_channels, hidden_channels, root_weight, bias))
            self.convs.append(HistSageLayer(hidden_channels, out_channels, root_weight, bias))
            
        self.reset_parameters()

    def _call_conv(self, layer_idx, x, ptr, idx, num_nodes, history_pack, frontier_rows, mapping_rows=None, is_mapping=False):
        return self.convs[layer_idx](
            x=x, ptr=ptr, idx=idx, num_nodes=num_nodes,
            history_pack=history_pack, frontier_rows=frontier_rows,
            stats=self.opstats, layer_idx=layer_idx, prev_rows=mapping_rows
        )
    
