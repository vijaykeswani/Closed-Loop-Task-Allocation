## Closed-Loop Models for Task Allocation

This folder contains code for the learning framework to train a *closed-loop* model for task allocation. The complete model and analysis are presented in the following paper:

**[Designing Closed-Loop Models for Task Allocation](https://arxiv.org/abs/2202.04718)** <br>
Vijay Keswani, L. Elisa Celis, Matthew Lease, Krishnaram Kenthapadi <br>
*Hybrid Human Artificial Intelligence (HHAI), 2023*

The files in this folder can be used for implementing the framework and replicating the results in the paper. The file description are provided below:

- `analysis_jigsaw_toxicity_dataset.ipynb`: This is the file with the primary code for the construction and training of a closed-loop task allocation model for the Jigsaw toxicity detection dataset.

The analysis for the toxicity detection requires data from two different sources noted below.

1. First is the main Jigsaw dataset. Please download the Jigsaw dataset from this link to execute the code below - https://github.com/Nihal2409/Jigsaw-Unintended-Bias-in-Toxicity-Classification/blob/master/Jigsaw_Unintended_Bias_in_Toxicity_Classification.ipynb

2. Second is the specialized rater pool data collected by Goyal et al. Please download this data from the following link - https://www.kaggle.com/datasets/google/jigsaw-specialized-rater-pools-dataset

Please run the `analysis_jigsaw_toxicity_dataset.ipynb` notebook to see how these datasets are being used and aggregated together.

- `analysis_synthetic_dataset.ipynb.ipynb`: This is the file with the code for the construction and training of a closed-loop task allocation model for a synthetic dataset. The main task here is to appropriately decipher all clusters in a given synthetic data using expert help.

- The folder `models` contains the main model architecture and training code for both the toxicity detection and synthetic cluster detection tasks.

