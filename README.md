# TAG-CP

**T**arget based **A**ttentive **G**raph neural network & **C**ombination **P**rediction (TAG-CP).

### Introduction

TAG-CP offers a novel computational model for synergistic drug combination through integrating drug-target relationship to represent small molecules with the framework of attentive graph neural network.

- To begin with, please get the codes with  ```git clone https://github.com/ZJunBio/TAG-CP.git``` or download the  **.zip** file with magnet https://github.com/ZJunBio/TAG-CP/archive/refs/heads/master.zip, and run the following scripts or commands in the **tag-cp** directory. 
- The NN directory saves the code used to build the Graph Attention Network (GAT) model and drug combination predicting model. The data_preprocess folder saves the codes for handling training or testing data.

### Environment Requirement

- The code has been tested  running  under Python 3.9.12. The key packages are as follows:

  - pytorch == 1.13.0

  - torch-geometric == 2.3.1

  - rdkit == 2023.3.1

  - pandas == 2.0.1

  - numpy == 1.24.3

  - scikit-learn == 1.3.0

- You can prepare the environment with conda, please try again if you failed to create the environment : 

  ``` shell 
  $ conda env create -n tag_cp -f requirement.yml
  $ conda activate tag_cp
  ```

  - If you have not installed **conda**，please refer to [Installing Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

###  Obtaining the graph embedding of drugs 

For now, this graph attention network (GAT) is a transductive  learning model and allowed embedding learning for 1362 drugs or compounds. We offer a  dictionary-structured file serialized with python pickle module  and you can directly use them for further study , or you can run the GAT model to generate the low-dimensional graph embedding. 

- With offered file, you can directly use the representation of compounds:

  ```python
  > import pickle
  > with open("data/drugs/graph_re.pickle", 'rb') as file: 
      cid_repre = pickle.load(file)
      #The key of a cid_repre record is the pubchem CID of a compound;
      #The value of a cid_repre record is the embedding of a compound;
  > #cid_repre[5311104] = array([0, 0, 0, 0.0024659 , 0...]
  ```

- Enter the ```jupyter-lab``` at terminal under conda environment, and run the GAT model within the notebook named **GAT_model.ipynb** located in **NN** directory, and the PubChem CIDs of drugs are saved in drug_list.txt

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

2. You can generate the synergy probability of prepared drug combinations with follow commands, the ```predict_pytorch.py```  and results are saved under **test** directory. 

    ```shell
    $ python data_preprocess/preprocess.py
    $ cd test
    $ python predict_pytorch.py lung_test.csv lung_prediction.csv
    $ # where the lung_prediction.csv is a user specified file including output.
    ```