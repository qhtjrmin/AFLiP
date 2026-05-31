import torch
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, to_undirected
from torch_geometric.data import Data

PIN_MEMORY = True
IS_GCN = False

# random split dataset
def randomsplit(dataset, val_ratio: float=0.10, test_ratio: float=0.2):
    def removerepeated(ei):
        ei = to_undirected(ei)
        ei = ei[:, ei[0]<ei[1]]
        return ei
    data = dataset[0]
    data.num_nodes = data.x.shape[0]
    data = train_test_split_edges(data, test_ratio, test_ratio)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio/test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[:, torch.randperm(data.val_pos_edge_index.shape[1])]
    split_edge['train']['edge'] = removerepeated(torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()
    return split_edge

def _extract_csr_from_adj(adj_t: SparseTensor):
    """
    SparseTensor -> CSR rowptr/col 추출
    - rowptr: [N+1]
    - col:    [E]
    """
    rowptr = adj_t.storage.rowptr()
    col    = adj_t.storage.col()
    edge   = adj_t.storage.value()

    rowptr = rowptr.to(torch.int64)
    col    = col.to(torch.int32)

    rowptr = rowptr.contiguous()
    col    = col.contiguous()
    return rowptr, col


def loaddataset(name: str, use_valedges_as_input: bool, load=None):
    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root="dataset", name=name)
        split_edge = randomsplit(dataset)
        data = dataset[0]
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
    else:
        dataset = PygLinkPropPredDataset(name=f'ogbl-{name}')
        split_edge = dataset.get_edge_split()
        data = dataset[0]
        edge_index = data.edge_index
        if not hasattr(data, 'num_nodes'):
            data.num_nodes = data.x.shape[0]
    data.edge_weight = None 
    
    data.orig_nid = torch.arange(data.num_nodes)
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))

    if IS_GCN: #add self-loop
        print("add self-loop")
        data.adj_t = data.adj_t.set_diag()
        
    data.adj_t = data.adj_t.to_symmetric().coalesce()

    data.ptr, data.idx = _extract_csr_from_adj(data.adj_t)

    #add degree
    data.deg = torch.diff(data.ptr)

    data.max_x = -1
    if name == "ppa":
        data.x = data.x.to(dtype=torch.float32)

    if load is not None:
        data.x = torch.load(load, map_location="cpu")
        data.max_x = -1

    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])

    # record data.input_dim
    data.input_dim = dataset.num_features
    print("data.input_dim:", data.input_dim)


    # Use training + validation edges for inference on test set.
    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    return data, split_edge

if __name__ == "__main__":
    loaddataset("Cora", False)
    loaddataset("Citeseer", False)
    loaddataset("Pubmed", False)
    loaddataset("ppa", False)
    loaddataset("collab", False)
    loaddataset("citation2", False)
    loaddataset("twitter", False)
    loaddataset("friendster", False)