# Diner

Code implementation for the anomaly detector Diner.


# Installation
### Requirements
* Python >= 3.6
* [Pytorch==1.10.2](https://pytorch.org/)

### Install packages
```
    pip intall -r requirements
```

### Quick Start
Run to check if the environment is ready
```
  python run.py
```

# Usage
We use part of GAIA dataset(refer to [GAIA-DataSet](https://github.com/CloudWise-OpenSource/GAIA-DataSet)) as demo example. 

## Data Preparation
```
# put your dataset under data/ directory with the same structure shown in the data/GAIA_periodic_data/

data
 |-GAIA_periodic_data
 | |-list.txt    # the kpi names, one kpi per line
 | |-data_n_train.csv   # training data
 | |-data_n_test.csv    # test data
 |-your_dataset
 | |-list.txt
 | |-{kpi_name}_train.csv
 | |-{kpi_name}_test.csv
 | ...

```

### Notices:
* The data files ({kpi_name}_train.csv and {kpi_name}_test.csv) should have three columns: "timestamps", "values", and "labels".
* The kpi list in list.txt is used to get all kpi names in the dataset.

## Run
```
    # using cpu
    python run.py
```
You can change running parameters in the run.py.

