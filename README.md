# Explainability-Driven Tree Ensemble Pruning Using Meta-Level Optimization

**Author:** Mohamed Amine Rekik 
**Supervisor:** Dr. Amal Saadallah 
**Institution:** TU Dortmund 

---

## 1. Abstract
TODO 

## 2. Datasets 

see https://github.com/treforevans/uci_datasets/tree/master 

Regression datasets from the [UCI machine learning repository](https://archive.ics.uci.edu) prepared for benchmarking studies with test-train splits.

# Installation

Install using pip :

```bash
python -m pip install git+https://github.com/treforevans/uci_datasets.git
```

you can choose one of the following datasets : juste update the dataset_name in the code  

| Dataset name     | Number of observations | Input dimension |
| :--------------- | ---------------------: | --------------: |
| `3droad`         |                 434874 |               3 |
| `autompg`        |                    392 |               7 |
| `bike`           |                  17379 |              17 |
| `challenger`     |                     23 |               4 |
| `concreteslump`  |                    103 |               7 |
| `energy`         |                    768 |               8 |
| `forest`         |                    517 |              12 |
| `houseelectric`  |                2049280 |              11 |
| `keggdirected`   |                  48827 |              20 |
| `kin40k`         |                  40000 |               8 |
| `parkinsons`     |                   5875 |              20 |
| `pol`            |                  15000 |              26 |
| `pumadyn32nm`    |                   8192 |              32 |
| `slice`          |                  53500 |             385 |
| `solar`          |                   1066 |              10 |
| `stock`          |                    536 |              11 |
| `yacht`          |                    308 |               6 |
| `airfoil`        |                   1503 |               5 |
| `autos`          |                    159 |              25 |
| `breastcancer`   |                    194 |              33 |
| `buzz`           |                 583250 |              77 |
| `concrete`       |                   1030 |               8 |
| `elevators`      |                  16599 |              18 |
| `fertility`      |                    100 |               9 |
| `gas`            |                   2565 |             128 |
| `housing`        |                    506 |              13 |
| `keggundirected` |                  63608 |              27 |
| `machine`        |                    209 |               7 |
| `pendulum`       |                    630 |               9 |
| `protein`        |                  45730 |               9 |
| `servo`          |                    167 |               4 |
| `skillcraft`     |                   3338 |              19 |
| `sml`            |                   4137 |              26 |
| `song`           |                 515345 |              90 |
| `tamielectric`   |                  45781 |               3 |
| `wine`           |                   1599 |              11 |


## 3. usage : 

Install the required packages :

```bash
python pip install -r requirements.txt 
```
and install the datasets  : 

```bash
python -m pip install git+https://github.com/treforevans/uci_datasets.git
```
then run the script  