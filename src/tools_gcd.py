import torch
import torch.nn as nn
import torch.nn.functional as F
from args import read_args
import numpy as np
import math

args = read_args()

class HetAgg(nn.Module):
    def __init__(self, args, feature_list,
                 gene_neigh_list_train, cell_neigh_list_train, drug_neigh_list_train,
                 gene_train_id_list, cell_train_id_list, drug_train_id_list):
        super(HetAgg, self).__init__()
        self.args = args
        self.embed_d = args.embed_d
        self.feature_list = feature_list  # [gene_matrix, cell_matrix, drug_matrix]
        self.gene_neigh_list_train = gene_neigh_list_train
        self.cell_neigh_list_train = cell_neigh_list_train
        self.drug_neigh_list_train = drug_neigh_list_train
        self.device = args.device

        # LSTMs
        self.gene_content_rnn = nn.LSTM(self.embed_d, self.embed_d // 2, 1, bidirectional=True).to(self.device)
        self.cell_content_rnn = nn.LSTM(self.embed_d, self.embed_d // 2, 1, bidirectional=True).to(self.device)
        self.drug_content_rnn = nn.LSTM(self.embed_d, self.embed_d // 2, 1, bidirectional=True).to(self.device)

        self.init_weights()

    def init_weights(self):
        """Initialize any linear layers, if needed."""
        pass

    def forward(self, triple_list_batch, triple_index):
        """
        The 'forward' method called by PyTorch when we do: self.model(batch_edges, triple_type).
        We'll parse (center, pos, neg) from 'triple_list_batch' and pass them to 'het_agg'.
        """
        c_ids = [x[0] for x in triple_list_batch]
        p_ids = [x[1] for x in triple_list_batch]
        n_ids = [x[2] for x in triple_list_batch]

        c_out, p_out, n_out = self.het_agg(triple_index, c_ids, p_ids, n_ids)
        return c_out, p_out, n_out

    def het_agg(self, triple_index, c_id_batch, pos_id_batch, neg_id_batch):
        """
        We use triple_index to figure out which node types (center, pos, neg) are Genes, Cells, or Drugs.
        Example setup:
          triple_index=0 => gene–gene
          triple_index=1 => gene–cell
          triple_index=2 => gene–drug
          triple_index=3 => cell–drug
        Then we call 'content_agg' with the correct node_type & offset.
        """
        if triple_index == 0:
            # G–G => (gene, gene, gene)
            c_agg = self.content_agg(c_id_batch, node_type=1) 
            p_agg = self.content_agg(pos_id_batch, node_type=1)
            n_agg = self.content_agg(neg_id_batch, node_type=1)
        elif triple_index == 1:
            # G–C => center=gene, pos=cell, neg=cell
            c_agg = self.content_agg(c_id_batch, node_type=1)
            p_agg = self.content_agg(pos_id_batch, node_type=2)
            n_agg = self.content_agg(neg_id_batch, node_type=2)
        elif triple_index == 2:
            # G–D => center=gene, pos=drug, neg=drug
            c_agg = self.content_agg(c_id_batch, node_type=1)
            p_agg = self.content_agg(pos_id_batch, node_type=3)
            n_agg = self.content_agg(neg_id_batch, node_type=3)
        elif triple_index == 3:
            # C–D => center=cell, pos=drug, neg=drug
            c_agg = self.content_agg(c_id_batch, node_type=2)
            p_agg = self.content_agg(pos_id_batch, node_type=3)
            n_agg = self.content_agg(neg_id_batch, node_type=3)
        else:
            # If you add more triple types, handle them here
            raise ValueError(f"Invalid triple_index: {triple_index}")

        return c_agg, p_agg, n_agg

    def content_agg(self, id_batch, node_type):
        """
        content_agg that can handle 'global IDs' for gene/cell/drug by subtracting offsets.
        node_type=1 => gene => local ID = global ID
        node_type=2 => cell => local ID = global ID - GENE_n
        node_type=3 => drug => local ID = global ID - (GENE_n + CELL_n)
        """
        device = self.device
        if node_type == 1:
            # Genes are [0..(G_n-1)]
            local_ids = id_batch
            feature_idx = 0
            rnn_module = self.gene_content_rnn
        elif node_type == 2:
            # Cells are [G_n..(G_n + C_n -1)] in global indexing
            local_ids = [x - self.args.GENE_n for x in id_batch]
            feature_idx = 1
            rnn_module = self.cell_content_rnn
        elif node_type == 3:
            # Drugs are [G_n + C_n..(G_n + C_n + D_n -1)] in global indexing
            local_ids = [x - (self.args.GENE_n + self.args.CELL_n) for x in id_batch]
            feature_idx = 2
            rnn_module = self.drug_content_rnn
        else:
            raise ValueError(f"Invalid node_type: {node_type}")

        # gather embeddings
        try:
            embed_list = [self.feature_list[feature_idx][lid] for lid in local_ids]
            content_embeddings = torch.stack(embed_list, dim=0).to(device)  # shape [batch_size, embed_d]
        except Exception as e:
            raise RuntimeError(f"content_agg: can't gather embeddings: {e}")

        # pass through LSTM
        content_embeddings = content_embeddings.unsqueeze(0)  # [1, batch_size, embed_d]
        lstm_out, _ = rnn_module(content_embeddings)          # [1, batch_size, embed_d]
        aggregated = torch.mean(lstm_out, dim=0)              # [batch_size, embed_d]

        return aggregated

    def cross_entropy_loss(self, c_batch, p_batch, n_batch): #  embed_d
        """
        standard -log sigmoid( pos - neg ) across last dimension
        c_batch, p_batch, n_batch shapes can be [T, B, embed_d] or [B, embed_d]
        we sum over dim=-1, then mean
        """
        # handle 3D or 2D
        sim_pos = torch.sum(c_batch * p_batch, dim=-1)
        sim_neg = torch.sum(c_batch * n_batch, dim=-1)
        # -log(sigmoid(pos-neg))
        loss = torch.mean(-torch.log(torch.sigmoid(sim_pos - sim_neg)))
        return loss

if __name__ == "__main__":
    import torch

    # Example test input
    id_batch = [0, 1, 2]  # example node indices
    node_type = 1         # 1: gene, 2: cell, 3: drug

    # Initialize the HetAgg class
    args = read_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Example features
    feature_list = [
        torch.rand(args.GENE_n, args.embed_d).to(device),
        torch.rand(args.CELL_n, args.embed_d).to(device),
        torch.rand(args.DRUG_n, args.embed_d).to(device),
    ]

    gene_neigh_list_train = [[i for i in range(10)] for _ in range(args.GENE_n)]
    cell_neigh_list_train = [[i for i in range(10)] for _ in range(args.CELL_n)]
    drug_neigh_list_train = [[i for i in range(3)] for _ in range(args.DRUG_n)]

    gene_train_id_list = list(range(args.GENE_n))
    cell_train_id_list = list(range(args.CELL_n))
    drug_train_id_list = list(range(args.DRUG_n))

    het_agg_model = HetAgg(
        args, feature_list,
        gene_neigh_list_train, cell_neigh_list_train, drug_neigh_list_train,
        gene_train_id_list, cell_train_id_list, drug_train_id_list
    ).to(device)

    # Quick check:
    triple_list_batch = [
        [0, 1, 2],  # (center, pos, neg) => (gene0, cell1, drug2)
        [3, 4, 5]
    ]
    triple_index = 0

    c_agg, p_agg, n_agg = het_agg_model(triple_list_batch, triple_index)
    print("Shapes:", c_agg.shape, p_agg.shape, n_agg.shape)

    # cross-entropy test
    loss_val = het_agg_model.cross_entropy_loss(c_agg, p_agg, n_agg)
    print("Loss value:", loss_val.item())
