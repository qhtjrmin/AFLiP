import torch
from torch import Tensor

@torch.no_grad()
def csr_neighbors_1hop(ptr: Tensor, idx: Tensor, seeds: Tensor) -> Tensor:
    if seeds.numel() == 0:
        return seeds
    seeds = seeds.to(dtype=torch.long)
    start = ptr[seeds].to(dtype=torch.long)
    end   = ptr[seeds + 1].to(dtype=torch.long)
    deg = (end - start)
    total = int(deg.sum().item())
    if total == 0:
        return seeds.new_empty((0,), dtype=torch.long)
    cumsum = torch.cumsum(deg, dim=0)
    base = (cumsum - deg)
    row_start_rep = torch.repeat_interleave(start, deg)
    base_rep = torch.repeat_interleave(base, deg)
    local = torch.arange(total, device=ptr.device, dtype=torch.long) - base_rep
    pos = row_start_rep + local
    nbr = idx[pos].to(dtype=torch.long)
    return nbr


@torch.no_grad()
def seeds_plus_1hop(ptr: torch.Tensor, idx: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    """
    seeds ∪ 1-hop(neighbors(seeds))
    """
    one_hop = csr_neighbors_1hop(ptr, idx, seeds)
    if one_hop.numel() == 0:
        return torch.unique(seeds.to(torch.long))
    return torch.unique(torch.cat([seeds.to(torch.long), one_hop], dim=0))

@torch.no_grad()
def expand_seeds_1hop(ptr: Tensor, idx: Tensor, seeds_u: Tensor, additional_seeds: Tensor = None) -> Tensor:
    if seeds_u.numel() == 0:
        return seeds_u
    
    start = ptr[seeds_u]
    end   = ptr[seeds_u + 1]
    deg   = end - start

    total = int(deg.sum().item())
    if total == 0:
        return seeds_u

    # Build positions into idx for all neighbor entries of seeds_u
    csum = torch.cumsum(deg, dim=0)

    # For each neighbor entry (length = total), which row in seeds_u it belongs to
    row = torch.repeat_interleave(
        torch.arange(deg.numel(), device=ptr.device, dtype=torch.long),
        deg
    )

    # local offset within each CSR segment: 0..deg[row]-1
    g = torch.arange(total, device=ptr.device, dtype=torch.long)
    local = g - (csum[row] - deg[row])

    pos = start[row] + local
    one_hop = idx[pos].to(torch.long)

    # Final: sorted unique
    if additional_seeds is not None:
        return torch.unique(torch.cat([seeds_u, one_hop, additional_seeds], dim=0))
    return torch.unique(torch.cat([seeds_u, one_hop], dim=0))