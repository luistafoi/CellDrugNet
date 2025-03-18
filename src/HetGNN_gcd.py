import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

from args import read_args
from tools_gcd import HetAgg  # aggregator code
from data_generator_rand_gcd import input_data  # your data generator with gene_to_gene_list, etc.

class HetGNN_GCD(object):
    """
    Example HetGNN pipeline that:
      - uses gene/cell/drug adjacency from data_generator_rand_seq.input_data
      - transforms self.input_data.gene_to_gene_list -> pos_gene_gene, etc.
      - trains on G–G, G–C, G–D, C–D edges
    """
    def __init__(self, args):
        super(HetGNN_GCD, self).__init__()
        self.args = args

        print("Initializing data generator ...")
        self.input_data = input_data(args)

        # ---------------------------------------------------------------------
        #  Convert adjacency lists -> pos_gene_gene, pos_gene_cell, etc.
        # ---------------------------------------------------------------------
        # Suppose you read lines like "geneID: neighborG1, neighborG2" into
        #   self.input_data.gene_to_gene_list
        # We build self.input_data.pos_gene_gene, etc.:

        self.input_data.pos_gene_gene = []
        for g_node, neigh_list in self.input_data.pos_gene_gene:
            for g2 in neigh_list:
                self.input_data.pos_gene_gene.append((g_node, g2))

        self.input_data.pos_gene_cell = []
        for g_node, neigh_list in self.input_data.pos_gene_cell:
            for c in neigh_list:
                self.input_data.pos_gene_cell.append((g_node, c))

        self.input_data.pos_gene_drug = []
        for g_node, neigh_list in self.input_data.pos_gene_drug:
            for d in neigh_list:
                self.input_data.pos_gene_drug.append((g_node, d))

        self.input_data.train_pos_cell_drug = []
        for c_node, neigh_list in self.input_data.pos_cell_drug:
            for d in neigh_list:
                self.input_data.train_pos_cell_drug.append((c_node, d))

        # If you have a separate test file, you can fill self.input_data.test_pos_cell_drug = []
        # in the data generator or here, similarly.

        # 1) Create random embeddings for each node type
        gene_embed = torch.randn(self.args.GENE_n, self.args.embed_d)
        cell_embed = torch.randn(self.args.CELL_n, self.args.embed_d)
        drug_embed = torch.randn(self.args.DRUG_n, self.args.embed_d)

        # 2) Move to GPU if available
        if self.args.cuda and torch.cuda.is_available():
            gene_embed = gene_embed.cuda()
            cell_embed = cell_embed.cuda()
            drug_embed = drug_embed.cuda()

        # 3) Make them trainable
        gene_embed = nn.Parameter(gene_embed, requires_grad=True)
        cell_embed = nn.Parameter(cell_embed, requires_grad=True)
        drug_embed = nn.Parameter(drug_embed, requires_grad=True)

        feature_list = [gene_embed, cell_embed, drug_embed]

        # 4) Build aggregator neighbor lists (already in self.input_data)
        gene_neigh_list_train = self.input_data.gene_neigh_list_train
        cell_neigh_list_train = self.input_data.cell_neigh_list_train
        drug_neigh_list_train = self.input_data.drug_neigh_list_train

        # 5) ID lists
        gene_train_id_list = list(range(self.args.GENE_n))
        cell_train_id_list = list(range(self.args.CELL_n))
        drug_train_id_list = list(range(self.args.DRUG_n))

        # 6) Create the HetAgg model
        self.model = HetAgg(
            args,
            feature_list,
            gene_neigh_list_train,
            cell_neigh_list_train,
            drug_neigh_list_train,
            gene_train_id_list,
            cell_train_id_list,
            drug_train_id_list
        )
        if self.args.cuda and torch.cuda.is_available():
            self.model.cuda()

        # 7) Create optimizer
        self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim = optim.Adam(self.parameters, lr=args.lr, weight_decay=0)
        self.model.init_weights()

    def build_triple_list(self):
        """
        Build a list of triple_list[triple_index],
         0 => gene–gene
         1 => gene–cell
         2 => gene–drug
         3 => cell–drug
        Each sublist: list of (center, pos, neg).
        We'll do negative sampling only for cell–drug edges, if you define train_neg_cell_drug, etc.
        """
        triple_list = [[] for _ in range(4)]

        # GENE–GENE edges
        for (g1, g2) in self.input_data.pos_gene_gene:
            triple_list[0].append((g1, g2, -1))

        # GENE–CELL edges
        for (g, c) in self.input_data.pos_gene_cell:
            triple_list[1].append((g, c, -1))

        # GENE–DRUG edges
        for (g, d) in self.input_data.pos_gene_drug:
            triple_list[2].append((g, d, -1))

        # CELL–DRUG edges (with negative sampling, if desired)
        # We'll do a naive approach: pick random negative from entire drug set
        # or from self.input_data.train_neg_cell_drug if you built that
        for (c, d) in self.input_data.train_pos_cell_drug:
            # If you haven't built a separate negative list, you can do:
            # neg_d = random.randint(0, self.args.DRUG_n - 1)
            #
            # or if you have self.input_data.train_neg_cell_drug:
            # neg_c, neg_d = random.choice(self.input_data.train_neg_cell_drug)
            # triple_list[3].append( (c, d, neg_d) )
            # ...
            neg_d = random.randint(0, self.args.DRUG_n - 1)
            triple_list[3].append((c, d, neg_d))
        

        # Debug printing:
        sizes = [len(t) for t in triple_list]
        print("[DEBUG] triple_list sizes:", sizes)

        for i, tlist in enumerate(triple_list):
            if len(tlist) > 0:
                print(f"triple_list[{i}] sample edges:", tlist[:5])

        return triple_list

    def save_embeddings(self):
        """
        Save learned embeddings for gene, cell, drug in CSV format.
        """
        model_path = self.args.embedding_save_path
        gene_out = self.model.feature_list[0].data.cpu().numpy()
        cell_out = self.model.feature_list[1].data.cpu().numpy()
        drug_out = self.model.feature_list[2].data.cpu().numpy()

        os.makedirs(model_path, exist_ok=True)

        # Genes
        with open(os.path.join(model_path, "gene_embeddings.csv"), "w") as f:
            for i in range(self.args.GENE_n):
                emb_str = ",".join(map(str, gene_out[i]))
                f.write(f"{i},{emb_str}\n")

        # Cells
        offset_cell = self.args.GENE_n
        with open(os.path.join(model_path, "cell_embeddings.csv"), "w") as f:
            for i in range(self.args.CELL_n):
                emb_str = ",".join(map(str, cell_out[i]))
                f.write(f"{i + offset_cell},{emb_str}\n")

        # Drugs
        offset_drug = self.args.GENE_n + self.args.CELL_n
        with open(os.path.join(model_path, "drug_embeddings.csv"), "w") as f:
            for i in range(self.args.DRUG_n):
                emb_str = ",".join(map(str, drug_out[i]))
                f.write(f"{i + offset_drug},{emb_str}\n")

        print("Embeddings saved to CSV.")

    def model_train(self):
        print("Starting HetGNN training ...")

        # If you have a checkpoint to load
        if self.args.checkpoint != '':
            print(f"Loading checkpoint: {self.args.checkpoint}")
            self.model.load_state_dict(torch.load(self.args.checkpoint))

        self.model.train()
        embed_d = self.args.embed_d

        for epoch in range(self.args.train_iter_n):
            print(f"Epoch {epoch} ...")

            # 1) Build triple list
            triple_list = self.build_triple_list()
            # triple_list[0]: G–G
            # triple_list[1]: G–C
            # triple_list[2]: G–D
            # triple_list[3]: C–D

            # 2) See how many edges in each sub-list
            triple_sizes = [len(t) for t in triple_list]
            min_len = min(triple_sizes) if any(triple_sizes) else 0

            # 3) Batch slicing
            batch_s = self.args.mini_batch_s
            batch_n = int(min_len / batch_s) if min_len > 0 else 0

            epoch_loss = 0.0
            num_batches = 0

            for b_idx in range(batch_n):
                c_outs = []
                p_outs = []
                n_outs = []

                for triple_type in range(4):
                    start_idx = b_idx * batch_s
                    end_idx = (b_idx + 1) * batch_s
                    batch_edges = triple_list[triple_type][start_idx:end_idx]
                    if len(batch_edges) == 0:
                        c_temp = torch.zeros(batch_s, embed_d, device=self.model.device)
                        p_temp = torch.zeros(batch_s, embed_d, device=self.model.device)
                        n_temp = torch.zeros(batch_s, embed_d, device=self.model.device)
                    else:
                        c_temp, p_temp, n_temp = self.model(batch_edges, triple_type)

                    c_outs.append(c_temp)
                    p_outs.append(p_temp)
                    n_outs.append(n_temp)

                # Stack them
                c_stack = torch.stack(c_outs, dim=0)  # shape [4, batch_s, embed_d]
                p_stack = torch.stack(p_outs, dim=0)
                n_stack = torch.stack(n_outs, dim=0)

                # Cross-entropy loss
                loss = self.model.cross_entropy_loss(c_stack, p_stack, n_stack, embed_d)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                epoch_loss += loss.item()
                num_batches += 1

                if b_idx % 100 == 0:
                    print(f"  [Batch {b_idx}] loss = {loss.item():.4f}")

            avg_loss = epoch_loss / (num_batches if num_batches > 0 else 1)
            print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

            # Save model checkpoint
            if epoch % self.args.save_model_freq == 0:
                ckpt_path = os.path.join(self.args.model_path, f"HetGNN_{epoch}.pt")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"[Epoch {epoch}] Model saved to {ckpt_path}.")

                # Save embeddings if desired
                self.save_embeddings()

        print("Training finished!")

if __name__ == "__main__":
    args = read_args()
    print("----- Arguments -----")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Fix random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    gcd_model = HetGNN_GCD(args)

    if args.train_test_label == 0:
        gcd_model.model_train()
    else:
        print("train_test_label != 0 => skipping training. (Implement test logic here)")


