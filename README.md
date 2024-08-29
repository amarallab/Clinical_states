# Clinical States

## Overview

This repository contains code for paper **Robust extraction of pneumonia-associated clinical states from electronic health records**.

## Project Structure

- **narrative/**: Jupyter notebooks documenting data curation, preprocessing, clustering, state transitions, and results visualization.
- **scripts/**: Python scripts for computing intermediate results, including silhouette coefficients and putative density peaks.
- **src/**: Python function modules used for data processing, clustering, and analysis.

- **README.md**: This file.

## Installation

To run the project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/amarallab/Clinical_states.git
   ```  

2. Navigate to the project directory:  

    ```bash
    cd Clinical_states
    ```  

3. Create and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

The [SCRIPT CarpeDiem Dataset](https://physionet.org/content/script-carpediem-dataset/1.1.0/) and the [MIMIC-IV dataset](https://physionet.org/content/mimiciv/3.0/) are available under open access with a Data Use Agreement. After downloading, place them in the `data/mimic_iv` and `data/carpediem` folders, respectively.

To derive clinical features, scores, and other relevant data, use the [MIMIC code repository](https://github.com/MIT-LCP/mimic-code). Save the derived tables in the `data/mimiciv_derived` folder.

Next, run the Jupyter notebooks `01_mimic_inclusion.ipynb` and `02_mimic_curation.ipynb` under /narrative folder. These notebooks will generate the included pneumonia cohort data and curate it, saving the results in the `data/mimiciv_included` and `data/curated` folders, respectively.

The Jupyter notebooks `03_cohort_characteristics.ipynb` and `04_feature_completeness.ipynb` in the `/narrative` folder characterize clinical features for the two datasets.

### Preprocessing

Run the `preprocess.py` script in the `scripts/` folder. This script computes silhouette coefficients, which measure the separation of ground truth extreme states (refer to the paper for details) for different preprocessing parameters. The results are stored in the `data/calculated/silhouette` folder.

Next, run the Jupyter notebook `05_preprocessing.ipynb` in the `/narrative` folder to visualize the preprocessing outcomes and select the optimal preprocessing parameters.

### Dimensionality Reduction

Run the Jupyter notebook `06_pca_nPC_loading.ipynb` in the `/narrative` folder. This notebook performs PCA dimensionality reduction with different numbers of PCs and feature sets, and inspects the top feature loadings for significant PCs. Refer to the paper for more details.

Then, run the Jupyter notebook `07_pca_auroc.ipynb` in the `/narrative` folder. This notebook trains support vector machines and tests them on in-distribution and out-of-distribution data, evaluating the PCA space in separating ground truth extreme states.

### Clustering

Run the `calc_dp_centers.py` and `calc_dp_centers_mimic_on_scriptPCA.py` scripts in the `scripts/` folder. These scripts calculate putative density peaks with different feature sets on bootstrapped samples and store the results in the `data/calculated/dp/240521_holdout` and `data/calculated/dp/mimic_on_scriptPCA` folders, respectively.

Next, run the Jupyter notebook `08_clustering.ipynb` in the `/narrative` folder. This notebook performs ensemble density peak clustering (eDPC) to identify clinical states with different feature sets/datasets and compares the final clustering solutions.

Optionally, you can run the `bench_clus_syn.py` script in the `scripts/` folder and the `08b_edpc_synthetic.ipynb` notebook in the `/narrative` folder, which demonstrate the performance of eDPC on synthetic data.

### State Transitions

Run the Jupyter notebook `09_state_transition.ipynb` in the `/narrative` folder. This notebook characterizes the clinical relevance and state transition properties for the SCRIPT train, test, and MIMIC-IV datasets.

   
