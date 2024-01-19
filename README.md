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


**Dataset Repositories**
- GWBG dataset: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.93.115104
- HOIP-HSE dataset: https://datadryad.org/stash/dataset/doi:10.5061/dryad.gq3rg
- MPS and MPL datasets: https://next-gen.materialsproject.org/
- EFE dataset: https://figshare.com/collections/Experimental_formation_enthalpies_for_intermetallic_phases_and_other_inorganic_compounds/3822835
- EU-TQT dataset: https://pubs.acs.org/doi/10.1021/acsami.9b16065
- TM dataset: https://github.com/ngs00/simd
- EBG dataset: https://pubs.acs.org/doi/10.1021/acs.jpclett.8b00124
- GTT dataset: https://figshare.com/articles/dataset/MAST-ML_Education_Datasets/7017254


## References
[1] Lee, J., Seko, A., Shitara, K., Nakayama, K., & Tanaka, I. (2016). Prediction model of band gap for inorganic compounds by combination of density functional theory calculations and machine learning techniques. Physical Review B, 93(11), 115104.

[2] Kim, C., Huan, T. D., Krishnan, S., & Ramprasad, R. (2017). A hybrid organic-inorganic perovskite dataset. Scientific Data, 4(1), 1-11.

[3] Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., ... & Persson, K. A. (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. APL materials, 1(1).
