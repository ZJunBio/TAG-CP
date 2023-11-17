# TAG-CP

This  framework named **T**arget based **A**ttentive **G**raph neural network & **C**ombination **P**rediction (TAG-CP).

### Introduction

Our work offers a novel molecular representing method which integrates drug-target relationship and introduces a new method featuring drug combination. There are two main steps to predict the interaction type of drug-drug-cancer cell line combinations (DDCs).

1. Obtaining the low-dimensional graph embedding of each compound;
2. Calculating the feature of drug combinations and cell lines as the input for predicting.

### Environment Requirement

- The code has been tested  running  under Python 3.9.12. The key packages are as follows:

  - pytorch == 1.13.0

  - torch-geometric == 2.3.1

  - rdkit == 2023.3.1

  - pandas == 2.0.1

  - numpy == 1.24.3

  - scikit-learn == 1.3.0

- You can prepare the environment with conda, please try again if you failed creating the environment : 

  ``` shell 
  conda env create -n tag_cp -f requirement.yml
  conda activate tag_cp
  ```

  - If you have not installed **conda**，please refer to [Installing Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

###  Obtaining the graph embedding of drugs 

For now, this grpah attention network (GAT) is a transductive  learning model and allowed embedding learning for 1362 drugs or compounds. We offer a  dictionary-structured file serialized with python pickle module  and you can directly use , or you can run the GAT model to generate the low-dimensional graph embedding. 

In the future work, we will developed the model frame of inductive learning for representation learning of more compounds.

- With offered file, you can directly use the representation of compounds:

  ```python
  > import pickle
  > with open("data/drugs/graph_re.pickle", 'rb') as file: 
      cid_repre = pickle.load(file)
      #The key of a cid_repre record is the pubchem CID of a compound;
      #The value of a cid_repre record is the embedding of a compound;
  > #cid_repre[5311104] = array([0, 0, 0, 0.0024659 , 0...]
  ```

- Enter the jupyter lab at terminal under conda environment, running the GAT model within jupyter notebook, and the list of drugs are saved in drug_list.txt

  ```shell
  $ jupyter-lab
  ```

### Predicting the drug combination

1. The input format of ```csv``` file.

   ```
   drug_row,drug_col,cell_line_name
   PubChem CID 1,PubChem CID 2,Cell's name in Cell Model Passports database
   PubChem CID 3,PubChem CID 4,Cell's name in Cell Model Passports database
   ```

2. You can generate the synergy probability of prepared drug combinations with follow commands. 

    ```shell
    $ python data_preprocess/preprocess.py
    $ cd test
    $ python predict_pytorch.py lung_test.csv lung_prediction.csv
    $ # where the lung_prediction.csv is a user specified file including output.
    ```