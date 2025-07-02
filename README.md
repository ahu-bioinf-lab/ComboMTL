# ComboMTL

# Introduction
Optimizing synergistic drug combination prediction by integrating side effect prediction within a multi-task learning framework

# Environment Requirement
The code has been tested running under Python 3.7. The required package are as follows:
* pytorch == 1.6.0
* numpy == 1.19.1
* sklearn == 0.23.2
* networkx == 2.5
* pandas == 1.1.2

# Installation
To install the required packages for running ComboMTL, please use the following command first
```bash
pip install -r requirements.txt
```
If you meet any problems when installing pytorch, please refer to [pytorch official website](https://pytorch.org/)



# Dataset
Datasets used in the paper:
* [Protein-Protein Interaction Network](https://www.nature.com/articles/s41467-019-09186-x#Sec23) is a comprehensive human interactome network.
* [Drug-protein Associations](https://www.nature.com/articles/s41467-019-09186-x#Sec23) are based on FDA-approved or clinically investigational drugs.
* [Cell-protein Associations](https://maayanlab.cloud/Harmonizome/dataset/CCLE+Cell+Line+Gene+Expression+Profiles) is harvested from the Cancer Cell Line Encyclopedia.
* [DrugCombDB](http://drugcombdb.denglab.org/main) is a database with the largest number of drug combinations to date.



  
