#!/usr/bin/env python3
"""
data_generator_rand_gcd.py

Example data generator that reads:
   train_gene_to_gene_neighbors.txt
   train_gene_to_cell_neighbors.txt
   train_gene_to_drug_neighbors.txt
   train_cell_to_drug_neighbors.txt

Each line in these files is in colon + comma format, e.g.:
   42:101,102,105

Meaning node 42 is connected to [101,102,105].
We store these edges as (42, 101), (42, 102), (42, 105) in the appropriate list.

We also optionally build neighbor lists for aggregator usage.
"""

import argparse
import random
import numpy as np
from collections import defaultdict


class input_data(object):
    def __init__(self, args):
        self.args = args
        random.seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)

        # Node counts
        self.G_n = args.GENE_n
        self.C_n = args.CELL_n
        self.D_n = args.DRUG_n

        # +++++++++++ Edge Lists +++++++++++
        # We'll store all POSITIVE edges for:
        #   gene–gene, gene–cell, gene–drug, cell–drug
        self.pos_gene_gene = []
        self.pos_gene_cell = []
        self.pos_gene_drug = []
        self.pos_cell_drug = []  # might be train + test, if you want to handle them differently

        # If you plan to do train/test splits or negative edges for cell–drug, you can define them here:
        self.train_pos_cell_drug = []
        self.test_pos_cell_drug  = []
        self.train_neg_cell_drug = []
        self.test_neg_cell_drug  = []

        # +++++++++++ Neighbor Lists (for aggregator) +++++++++++
        self.gene_neigh_list_train = defaultdict(list)
        self.cell_neigh_list_train = defaultdict(list)
        self.drug_neigh_list_train = defaultdict(list)

        # 1) Load adjacency from "train_gene_to_gene_neighbors.txt"
        #    using colon+comma format
        self._load_gene_gene_neighbors_colon(
            filepath=self.args.data_path + "train_gene_to_gene_neighbors.txt"
        )

        # 2) Load adjacency from "train_gene_to_cell_neighbors.txt"
        self._load_gene_cell_neighbors_colon(
            filepath=self.args.data_path + "train_gene_to_cell_neighbors.txt"
        )

        # 3) Load adjacency from "train_gene_to_drug_neighbors.txt"
        self._load_gene_drug_neighbors_colon(
            filepath=self.args.data_path + "train_gene_to_drug_neighbors.txt"
        )

        # 4) Load adjacency from "train_cell_to_drug_neighbors.txt"
        #    If you have a separate test set, do that as well
        self._load_cell_drug_neighbors_colon(
            filepath=self.args.data_path + "train_cell_to_drug_neighbors.txt"
        )

        # Example: If you want to treat "train_cell_to_drug_neighbors.txt" as
        # the training portion, and have a "test_cell_to_drug_neighbors.txt" file, do:
        self._load_cell_drug_neighbors_colon(
            filepath=self.args.data_path + "test_cell_to_drug_neighbors.txt",
            is_train=False
        )

        # 5) If you want to build negative edges, do it here
        #    or you can do negative sampling on the fly in HetGNN.py

        # 6) Optionally build aggregator neighbor lists
        self._build_neighbor_lists()

        # Print stats
        self._print_stats()


    # ========================================================
    #   LOADING GENE-GENE
    # ========================================================
    def _load_gene_gene_neighbors_colon(self, filepath):
        """
        Expects lines like:
           geneID: neighborGene1,neighborGene2,...
        We'll parse them, then store each adjacency as (geneID, neigh).
        """
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if ":" not in line:
                        # skip malformed
                        print(f"Skipping line (no colon found): {line}")
                        continue
                    g_str, neigh_str = line.split(":", 1)  # split once
                    g = int(g_str)

                    # parse comma-separated neighbors
                    neigh_str = neigh_str.strip()
                    if not neigh_str:
                        # no neighbors
                        continue
                    neighbors = [x.strip() for x in neigh_str.split(",") if x.strip()]
                    for neigh in neighbors:
                        gg = int(neigh)
                        self.pos_gene_gene.append((g, gg))

        except FileNotFoundError:
            print(f"File not found: {filepath} (Skipping)")

    # ========================================================
    #   LOADING GENE-CELL
    # ========================================================
    def _load_gene_cell_neighbors_colon(self, filepath):
        """
        Expects lines like:
           geneID: cell1,cell2,...
        We'll parse them, store as (geneID, cellID).
        """
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if ":" not in line:
                        print(f"Skipping line (no colon found): {line}")
                        continue
                    g_str, neigh_str = line.split(":", 1)
                    g = int(g_str)

                    neigh_str = neigh_str.strip()
                    if not neigh_str:
                        continue
                    neighbors = [x.strip() for x in neigh_str.split(",") if x.strip()]
                    for c_str in neighbors:
                        c = int(c_str)
                        self.pos_gene_cell.append((g, c))

        except FileNotFoundError:
            print(f"File not found: {filepath} (Skipping)")

    # ========================================================
    #   LOADING GENE-DRUG
    # ========================================================
    def _load_gene_drug_neighbors_colon(self, filepath):
        """
        Expects lines like:
           geneID: drug1,drug2,...
        We'll parse them, store as (geneID, drugID).
        """
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if ":" not in line:
                        print(f"Skipping line (no colon found): {line}")
                        continue
                    g_str, neigh_str = line.split(":", 1)
                    g = int(g_str)

                    neigh_str = neigh_str.strip()
                    if not neigh_str:
                        continue
                    neighbors = [x.strip() for x in neigh_str.split(",") if x.strip()]
                    for d_str in neighbors:
                        d = int(d_str)
                        self.pos_gene_drug.append((g, d))

        except FileNotFoundError:
            print(f"File not found: {filepath} (Skipping)")

    # ========================================================
    #   LOADING CELL-DRUG
    # ========================================================
    def _load_cell_drug_neighbors_colon(self, filepath, is_train=True):
        """
        Expects lines like:
           cellID: drug1,drug2,...
        We'll parse them, store as (cellID, drugID).
        If is_train=True => put in self.train_pos_cell_drug
        else => put in self.test_pos_cell_drug
        or if you want to unify, store in self.pos_cell_drug
        """
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if ":" not in line:
                        print(f"Skipping line (no colon found): {line}")
                        continue
                    c_str, neigh_str = line.split(":", 1)
                    c = int(c_str)

                    neigh_str = neigh_str.strip()
                    if not neigh_str:
                        continue
                    neighbors = [x.strip() for x in neigh_str.split(",") if x.strip()]
                    for d_str in neighbors:
                        d = int(d_str)
                        if is_train:
                            self.train_pos_cell_drug.append((c, d))
                        else:
                            self.test_pos_cell_drug.append((c, d))

        except FileNotFoundError:
            print(f"File not found: {filepath} (Skipping)")

    # ========================================================
    #   BUILD NEIGHBOR LISTS (for aggregator)
    # ========================================================
    def _build_neighbor_lists(self):
        """
        If your aggregator uses gene/cell/drug neighbor lists, fill them here.
        We'll just add each positive edge in a one-sided manner as an example.
        If you want symmetrical aggregator (both directions), also add reverse.
        """
        # 1) GENE-GENE
        for (g1, g2) in self.pos_gene_gene:
            self.gene_neigh_list_train[g1].append(g2)
            # if you want symmetrical, also do: self.gene_neigh_list_train[g2].append(g1)

        # 2) GENE-CELL
        for (g, c) in self.pos_gene_cell:
            self.gene_neigh_list_train[g].append(c)
            # self.cell_neigh_list_train[c].append(g)  # if aggregator uses that

        # 3) GENE-DRUG
        for (g, d) in self.pos_gene_drug:
            self.gene_neigh_list_train[g].append(d)
            # self.drug_neigh_list_train[d].append(g)

        # 4) CELL-DRUG (train only, for aggregator, if you want)
        for (c, d) in self.train_pos_cell_drug:
            self.cell_neigh_list_train[c].append(d)
            self.drug_neigh_list_train[d].append(c)

    # ========================================================
    #   PRINT STATS
    # ========================================================
    def _print_stats(self):
        print("\n===== Data Generator Stats =====")
        print(f"Gene–Gene edges: {len(self.pos_gene_gene)}")
        print(f"Gene–Cell edges: {len(self.pos_gene_cell)}")
        print(f"Gene–Drug edges: {len(self.pos_gene_drug)}")
        print(f"Train Cell–Drug edges (pos): {len(self.train_pos_cell_drug)}")
        print(f"Test  Cell–Drug edges (pos): {len(self.test_pos_cell_drug)}")
        # If you do negative sampling, you can show it here
        print("Sample G–G:", self.pos_gene_gene[:3])
        print("Sample G–C:", self.pos_gene_cell[:3])
        print("Sample G–D:", self.pos_gene_drug[:3])
        print("Sample Train C–D:", self.train_pos_cell_drug[:3])
        print("Sample Test  C–D:", self.test_pos_cell_drug[:3])
        print("================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Generator for G–C–D with colon+comma format")
    parser.add_argument("--data_path", type=str, default="../data/", help="Path to data")
    parser.add_argument("--GENE_n", type=int, default=19978, help="Number of gene nodes")
    parser.add_argument("--CELL_n", type=int, default=1751, help="Number of cell nodes")
    parser.add_argument("--DRUG_n", type=int, default=5474, help="Number of drug nodes")
    parser.add_argument("--embed_d", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    # Additional HetGNN hyperparams:
    parser.add_argument("--batch_s", type=int, default=20000)
    parser.add_argument("--mini_batch_s", type=int, default=200)
    parser.add_argument("--train_iter_n", type=int, default=50)
    parser.add_argument("--save_model_freq", type=int, default=2)
    parser.add_argument("--cuda", type=int, default=1)
    args = parser.parse_args()

    data_obj = input_data(args)



