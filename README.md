# CellDrugNet: Graph Neural Network Framework for Drug Response Prediction

CellDrugNet is a GNN-based framework that integrates multi-relational data—including genomic, transcriptomic, and molecular data—to improve drug response prediction. This project aims to expand precision oncology by accurately predicting drug efficacy for new drugs and patients using graph-based methods.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Installation and Environment Setup](#installation-and-environment-setup)
- [Data](#data-availability)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Precision oncology is limited by the small subset of actionable genes used in treatment. CellDrugNet addresses this by using graph neural networks to integrate complex relationships among genes, drugs, and cell lines, providing insights for effective drug matching.

## Methodology
CellDrugNet constructs a heterogeneous graph with nodes representing genes, drugs, and cell lines, and edges capturing known biological interactions (e.g., protein-protein interactions, drug-gene mechanisms). Our model uses these connections to perform link prediction, indicating potential drug efficacy for specific cell lines.

## Installation and Environment Setup

Follow these steps to set up the environment for the CellDrugNet project:

1. **Clone the repository**:
   
   ```bash
   git clone https://github.com/luistafoi/CellDrugNet.git
   ```
2. *Navigate to the project directory:**:
   
   ```bash
   cd CellDrugNet
   ```

3. **Create the Conda environment:**:
   
   ```bash
   conda env create -f environment.yml
   ```

4. **Activate the Environment:**:
   
   ```bash
   conda activate CellDrugNet
   ```
5. **Download the Data**:

- Refer to the [Data](#data-availability) section for instructions on downloading the necessary datasets.
- After downloading, place the data files in the data/ folder as described.

This setup will prepare the environment for exploring and contributing to the code. Future updates will include detailed instructions for training and evaluating the model.


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
    |      ├── Embeddings/
    |      |      ├── gene_embeddings_512.csv
    |      |      ├── drug_embeddings_chemberta_512.csv
    |      |      └── cell_embeddings_512.csv
    |      ├── Datasets/
    |             ├── genetocell.csv
    |             ├── drug_SMILES.csv
    |             ├── gene_gene_association.csv
    |             └── genetodrug.csv
    ```

Please contact us if you encounter issues accessing the data.

## Usage

This project is currently under development, and training or evaluation scripts are not fully functional. However, you can explore the code and set up the environment as follows:

1. **Set Up the Environment**:
   Follow the instructions in the [Environment Setup](#installation-and-environment-setup) section to create the necessary Conda environment.

2. **Explore the Code**:
   - The main code files are located in the `src/` directory. 
   - `CellDrugNet_LM.py`: The primary Python file where the model architecture and functions are defined.
   - `CellDrugNet_LM.ipynb`: A Jupyter notebook that can be used for interactive exploration and testing of the code.

3. **Preparing for Model Training**:
   Once the model is ready for training, we will provide a detailed guide here. 

4. **Configuration**:
   If you plan to modify configurations, place them in a `configs/` directory with a YAML file (e.g., `config.yaml`). Detailed configuration instructions will be added once training functionality is implemented.

Stay tuned for updates, as we will be adding training and evaluation commands as the code develops!

### Results

CellDrugNet demonstrates significant improvements in predicting drug response. See the Results section in our paper for detailed performance metrics and ablation studies.

### Contributing

We welcome contributions! Please see our CONTRIBUTING.md for details.

License

This project is NOT YET licensed.
