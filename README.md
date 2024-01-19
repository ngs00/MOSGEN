# Learning M-Order Spectrum Graphs to Identify Unknown Chemical Compounds from Infrared Spectroscopy Data

## Abstract
Infrared (IR) spectrum analysis is an efficient and generally applicable method for identifying unknown chemical compounds.
However, the analysis quality of the IR spectrum essentially depends on a labor-intensive manual search of human experts.
We propose a graph-based machine learning approach for automatically identifying unknown chemical compounds from the input IR spectra without the manual analysis of human experts.
To this end, we define a graph representation of unstructured spectrum data and devise a sequence model based on graph neural networks.
We evaluated the proposed method in three common tasks of the IR spectrum analysis: material class classification, functional group detection, and compound identification.
In these experiments, the proposed method achieved state-of-the-art prediction accuracy.
In particular, the proposed method showed a prediction accuracy of 41.82%-85.10% in identifying unknown chemical compounds from their IR spectra, which is one of the most challenging tasks in chemical analysis.


## Run
- ``material_class_classification.py``: Train and evaluate MOSGEN in the material class classification task.
- ``functional_group_detection.py``: Train and evaluate MOSGEN in the functional group detection tasks.
- ``compound_identification.py``: Train and evaluate MOSGEN in the compound identification task.


## Datasets
We trained MOSGEN on an IR spectrum dataset of real-world 1,238 chemical compounds from 9 different material classes.
The IR spectra of the 1,238 chemical compounds were collected from [the Infrared and Raman User Group (IRUG) database](http://www.irug.org).

## Required Packages
- RDKit: https://www.rdkit.org/
- PyTorch: https://pytorch.org/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/
