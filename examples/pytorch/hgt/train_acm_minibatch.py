#!/usr/bin/env python
# coding: utf-8

import argparse
import math
import urllib.request

import numpy as np
import scipy.io
from model_minibatch import *

import dgl

torch.manual_seed(0)
data_url = "https://data.dgl.ai/dataset/ACM.mat"
data_file_path = "/tmp/ACM.mat"

urllib.request.urlretrieve(data_url, data_file_path)
data = scipy.io.loadmat(data_file_path)


parser = argparse.ArgumentParser(
    description="Training GNN on ogbn-products benchmark"
)


parser.add_argument("--n_epoch", type=int, default=200)
parser.add_argument("--n_hid", type=int, default=256)
parser.add_argument("--n_inp", type=int, default=256)
parser.add_argument("--clip", type=int, default=1.0)
parser.add_argument("--max_lr", type=float, default=1e-3)

args = parser.parse_args()

from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch_sparse

def rename_edge_dict(edge_dict):
    rename_edege_dict = {}
    for i in edge_dict.keys():
        rename_edege_dict[tuple(i.split('__'))] = edge_dict[i]
    return rename_edege_dict

class HGTSampler(dgl.dataloading.Sampler):
    def __init__(self, num_samples : Union[int, Dict[str, int]], num_layer: int):
        super().__init__()
        self.num_samples = num_samples
        self.num_layer = num_layer
        #{key: num_samples for key in g.ntypes}
        #self.num_hops = max([len(v) for v in num_samples.values()])

    def sample(self, g, seed_nodes):
        output_nodes = seed_nodes
        subgs = []
        
        if isinstance(self.num_samples, int):
            num_samples = [self.num_samples] * 1 # layer
            num_samples = {key: num_samples for key in g.ntypes}

        num_hops = max([len(v) for v in num_samples.values()])
        for i in range(self.num_layer):
            #print(f"layer: {i}")
            colptr_dict = {}
            row_dict = {}
            # filter only inbound edges
            g_tmp = dgl.in_subgraph(g, seed_nodes)

            # Convert DGL graph to CSC metrics
            for i in g_tmp.canonical_etypes:
                e_name = '__'.join(i)
                colptr_dict[e_name] = g_tmp.adj_sparse('csc', etype=i)[0]
                row_dict[e_name] = g_tmp.adj_sparse('csc', etype=i)[1]

            # Sample a fixed number of neighbors of the current seed nodes.            
            node_dict, row_dict, col_dict, edge_dict = torch.ops.torch_sparse.hgt_sample(colptr_dict, row_dict, seed_nodes, num_samples, num_hops)
            sg = dgl.edge_subgraph(g_tmp, rename_edge_dict(edge_dict), relabel_nodes=False)
   
            # Convert this subgraph to a message flow graph.
            sg = dgl.to_block(sg, seed_nodes)            
            seed_nodes = sg.srcdata[dgl.NID]
            subgs.insert(0, sg)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, subgs
    

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def train(model, G):
    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)
    train_step = torch.tensor(0)
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        logits = model(G, "paper")
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        if epoch % 5 == 0:
            model.eval()
            logits = model(G, "paper")
            pred = logits.argmax(1).cpu()
            train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
            val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
            test_acc = (pred[test_idx] == labels[test_idx]).float().mean()
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            print(
                "Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)"
                % (
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss.item(),
                    train_acc.item(),
                    val_acc.item(),
                    best_val_acc.item(),
                    test_acc.item(),
                    best_test_acc.item(),
                )
            )

def train2(model, G):
    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)
    train_step = torch.tensor(0)
    n_layers = 2
    fanout = 16
    batch_size = 1024
    print(f"setting fanout as {fanout}")
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [fanout] * n_layers
        )
    #sampler = HGTSampler(fanout, n_layers)
    #train_idx = train_idx.to(device)
    #test_idx = test_idx.to(device)
    loader = dgl.dataloading.DataLoader(
            G,
            {"paper": train_idx.to(device)},
            sampler,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            device=device,
            #use_uva=True
        )
    sampler2 = dgl.dataloading.MultiLayerNeighborSampler(
            [fanout] * n_layers
        )
    #sampler2 = HGTSampler(fanout, n_layers)
    loader2 = dgl.dataloading.DataLoader(
            G,
            {"paper": test_idx.to(device)},
            sampler,
            batch_size=batch_size*100,
            shuffle=True,
            num_workers=0,
            device=device,
            #use_uva=True
        )
    sampler3 = dgl.dataloading.MultiLayerNeighborSampler(
            [fanout] * n_layers
        )
    #sampler3 = HGTSampler(fanout, n_layers)
    loader3 = dgl.dataloading.DataLoader(
            G,
            {"paper": val_idx.to(device)},
            sampler,
            batch_size=batch_size*100,
            shuffle=True,
            num_workers=0,
            device=device,
            #use_uva=True
        )
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        for i, (input_nodes, seeds, blocks) in enumerate(loader):
            blocks = [blk.to(device2) for blk in blocks]
            logits = model(blocks, "paper")
            #print(seeds)
            #print(len(seeds['paper']))
            # The loss is computed only for labeled nodes.
            loss = F.cross_entropy(logits, labels[seeds['paper']].to(device2))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        
        if epoch % 5 == 0:
            model.eval()
            for i, (input_nodes, seeds, blocks) in enumerate(loader):
                blocks = [blk.to(device2) for blk in blocks]
                logits = model(blocks, "paper")
                pred = logits.argmax(1).cpu()
                train_acc = (pred == labels[seeds['paper']]).float().mean()
                
            for i, (input_nodes, seeds, blocks) in enumerate(loader2):
                blocks = [blk.to(device2) for blk in blocks]
                logits = model(blocks, "paper")
                pred = logits.argmax(1).cpu()
                test_acc = (pred == labels[seeds['paper']]).float().mean()
            
            for i, (input_nodes, seeds, blocks) in enumerate(loader3):
                blocks = [blk.to(device2) for blk in blocks]
                logits = model(blocks, "paper")
                pred = logits.argmax(1).cpu()
                val_acc = (pred == labels[seeds['paper']]).float().mean()

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            print(
                "Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)"
                % (
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss.item(),
                    train_acc.item(),
                    val_acc.item(),
                    best_val_acc.item(),
                    test_acc.item(),
                    best_test_acc.item(),
                )
            )


            
            

#device = torch.device("cpu")
device = torch.device("cuda:0")
device2 = torch.device("cuda:0")

G = dgl.heterograph(
    {
        ("paper", "written-by", "author"): data["PvsA"].nonzero(),
        ("author", "writing", "paper"): data["PvsA"].transpose().nonzero(),
        ("paper", "citing", "paper"): data["PvsP"].nonzero(),
        ("paper", "cited", "paper"): data["PvsP"].transpose().nonzero(),
        ("paper", "is-about", "subject"): data["PvsL"].nonzero(),
        ("subject", "has", "paper"): data["PvsL"].transpose().nonzero(),
    }
)
print(G)

pvc = data["PvsC"].tocsr()
p_selected = pvc.tocoo()
# generate labels
labels = pvc.indices
labels = torch.tensor(labels).long()

# generate train/val/test split
pid = p_selected.row
shuffle = np.random.permutation(pid)
train_idx = torch.tensor(shuffle[0:800]).long()
val_idx = torch.tensor(shuffle[800:900]).long()
test_idx = torch.tensor(shuffle[900:]).long()

node_dict = {}
edge_dict = {}
for ntype in G.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in G.etypes:
    edge_dict[etype] = len(edge_dict)
    G.edges[etype].data["id"] = (
        torch.ones(G.number_of_edges(etype), dtype=torch.long)
        * edge_dict[etype]
    )

#     Random initialize input feature
for ntype in G.ntypes:
    emb = nn.Parameter(
        torch.Tensor(G.number_of_nodes(ntype), 256), requires_grad=False
    )
    nn.init.xavier_uniform_(emb)
    G.nodes[ntype].data["inp"] = emb

G = G.to(device)


model = HGT2(
    G,
    node_dict,
    edge_dict,
    n_inp=args.n_inp,
    n_hid=args.n_hid,
    n_out=labels.max().item() + 1,
    n_layers=2,
    n_heads=4,
    use_norm=True,
).to(device2)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, total_steps=args.n_epoch, max_lr=args.max_lr
)
print("----------------")
print("Minibatch Training with HGT")
train2(model, G)


model = HGT(
    G,
    node_dict,
    edge_dict,
    n_inp=args.n_inp,
    n_hid=args.n_hid,
    n_out=labels.max().item() + 1,
    n_layers=2,
    n_heads=4,
    use_norm=True,
).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, total_steps=args.n_epoch, max_lr=args.max_lr
)
print("----------------")
print("Fullbatch Training with HGT")
train(model, G)




model = HeteroRGCN(
    G,
    in_size=args.n_inp,
    hidden_size=args.n_hid,
    out_size=labels.max().item() + 1,
).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, total_steps=args.n_epoch, max_lr=args.max_lr
)
#print("Training RGCN with #param: %d" % (get_n_params(model)))
print("----------------")
print("Fullbatch Training with RGCN")
train(model, G)


model = HGT(
    G,
    node_dict,
    edge_dict,
    n_inp=args.n_inp,
    n_hid=args.n_hid,
    n_out=labels.max().item() + 1,
    n_layers=0,
    n_heads=4,
).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, total_steps=args.n_epoch, max_lr=args.max_lr
)
print("----------------")
print("Fullbatch Training with MLP")
train(model, G)
