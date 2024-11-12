# CellDrugNet: Graph Neural Network Framework for Drug Response Prediction

CellDrugNet is a GNN-based framework that integrates multi-relational data—including genomic, transcriptomic, and molecular data—to improve drug response prediction. This project aims to expand precision oncology by accurately predicting drug efficacy for new drugs and patients using graph-based methods.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Installation and Environment Setup](#installation-and-environment-setup)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
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
