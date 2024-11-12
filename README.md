# CellDrugNet: Graph Neural Network Framework for Drug Response Prediction

CellDrugNet is a GNN-based framework that integrates multi-relational data—including genomic, transcriptomic, and molecular data—to improve drug response prediction. This project aims to expand precision oncology by accurately predicting drug efficacy for new drugs and patients using graph-based methods.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Installation and Environment Setup](#installation-and-environment-setup)
- [Usage](#usage)
- [Data Availabilty](#data-availability)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Precision oncology is limited by the small subset of actionable genes used in treatment. CellDrugNet addresses this by using graph neural networks to integrate complex relationships among genes, drugs, and cell lines, providing insights for effective drug matching.

## Methodology
CellDrugNet constructs a heterogeneous graph with nodes representing genes, drugs, and cell lines, and edges capturing known biological interactions (e.g., protein-protein interactions, drug-gene mechanisms). Our model uses these connections to perform link prediction, indicating potential drug efficacy for specific cell lines.

## Installation and Environment Setup

To set up the environment, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/luistafoi/CellDrugNet.git

2. **Navigate to the project directory:**:
   ```bash
   cd CellDrugNet

3. **Create the Conda environment::**:
   ```bash
   conda env create -f environment.yml

4. **Activate the Environment::**:
   ```bash
   conda activate CellDrugNet

**Note:** Ensure Conda is installed and up to date.

## Data Availability

Due to storage limitations, the processed datasets and embeddings used for CellDrugNet are hosted externally. Below is a list of the data sources, processing steps, and instructions for accessing the final data.

### Data Sources
- **Gene Expression Data**: DepMap (https://depmap.org)
- **Drug Molecular Structures**: PubChem (https://pubchem.ncbi.nlm.nih.gov)
- **Protein-Protein Interactions (PPI)**: UniProt (https://www.uniprot.org)

### Processed Datasets and Embeddings
The final datasets and embeddings are available for download at:

[Google Drive link to processed data]([https://drive.google.com/drive/folders/your_shared_folder_link](https://drive.google.com/drive/folders/156-HcL9vjeLbM8ZMqwmg0mGyAeo_qynl?usp=sharing))

**Instructions**:
1. Download the files from the link above.
2. Place the datasets in the `data/` directory within the project, following this structure:

    ```
    ├── data/
    │   ├── cell_lines.csv
    │   ├── drugs.csv
    │   ├── gene_embeddings.csv
    │   └── other_data_files.csv
    ```

Please contact us if you encounter issues accessing the data.
