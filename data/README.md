## Data Availability and Setup

To use CellDrugNet, you will need to download the processed datasets and embeddings, as they are too large to be stored directly in the repository.

### Step 1: Download the Data

The processed datasets are hosted externally. You can download the necessary files from the following link:

- [Google Drive link to processed data](https://drive.google.com/drive/folders/156-HcL9vjeLbM8ZMqwmg0mGyAeo_qynl?usp=sharing)

### Step 2: Organize the Data

Once downloaded, please organize the files in a `data/` directory in the project’s root folder. The folder structure should look like this:

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
    ├── env/
    │   └── environment.yml           # Conda environment setup file
    ├── src/
    │   ├── CellDrugNet_LM.py         # Main Python file
    │   ├── CellDrugNet_LM.ipynb      # Jupyter notebook
    └── README.md                     # Main README with setup instructions
 ```

### Important Note
Ensure the files are named exactly as indicated above and are in the correct directory. The code depends on this structure to locate and load the data correctly.

### Data Sources
For reference, here are the sources of the original data:
- **Gene Expression Data**: DepMap (https://depmap.org)
- **Drug Molecular Structures**: PubChem (https://pubchem.ncbi.nlm.nih.gov)
- **Protein-Protein Interactions (PPI)**: UniProt (https://www.uniprot.org)

If you have any issues accessing the data, please contact us.
