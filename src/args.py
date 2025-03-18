import argparse
import torch 

def read_args():
    parser = argparse.ArgumentParser(description="HetGNN for Gene-Cell-Drug data")

    # Paths
    parser.add_argument('--data_path', type=str, default='../data/',
                        help='Path to data directory')
    parser.add_argument('--model_path', type=str, default='../data/model_save/',
                        help='Path to save model')
    parser.add_argument('--embedding_save_path', type=str, default='../data/embedding_save/',
                        help='Path to save learned node embeddings')

    # Node counts
    parser.add_argument('--GENE_n', type=int, default=19978,
                        help='Number of gene nodes')
    parser.add_argument('--CELL_n', type=int, default=1751,
                        help='Number of cell nodes')
    parser.add_argument('--DRUG_n', type=int, default=5474,
                        help='Number of drug nodes')

    # Feature dimensions
    parser.add_argument('--in_f_d', type=int, default=512,
                        help='Input feature dimension')
    parser.add_argument('--embed_d', type=int, default=512,
                        help='Embedding dimension')

    # Training settings
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_s', type=int, default=20000,
                        help='Batch size') # 500
    parser.add_argument('--mini_batch_s', type=int, default=200,
                        help='Mini batch size')
    parser.add_argument('--train_iter_n', type=int, default=50,
                        help='Max number of training iterations')
    parser.add_argument('--save_model_freq', type=int, default=2,
                        help='Number of iterations to save model')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Random walk settings
    parser.add_argument('--walk_n', type=int, default=10, help='Number of walks per root node')
    parser.add_argument('--walk_L', type=int, default=10, help='Length of each walk')
    parser.add_argument('--window', type=int, default=5, help='Window size for relation extraction')

    # Training or testing modes
    parser.add_argument('--train_test_label', type=int, default=0,
                        help='train/test label: 0 - train, 1 - test, 2 - code test/generate negative ids for evaluation')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to checkpoint file for model loading')

    # Hardware settings
    parser.add_argument('--cuda', type=int, default=1,
                        help='Use GPU: 1 for True, 0 for False')

    args = parser.parse_args()

    # Add device attribute
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    return args

