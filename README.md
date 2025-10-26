# PyTorch Finance Starter Code

This repository provides a starter code for a finance deep learning project using PyTorch.

The repository is organized as follows.

```
├── 📂 pipeline/
│   ├── 📂 dataset/
│   │   ├── 📂 raw/
│   │   │   ├── 📄 spy_YYYY_MM.csv
│   │   ├── 📄 download.ipynb
│   │   └── 📄 eda.ipynb
│   ├── 📄 __init__.py
│   ├── 📄 data.py
│   ├── 📄 model.py
│   ├── 📄 train.py
│   └── 📄 utils.py
├── 📄 .gitignore
├── 📄 requirements.txt
├── 📄 tuning.py
└── 📄 main.py
```

- `pipeline` is a custom package for handling the different components of model training.

    - `pipeline/dataset/raw/` is the directory for placing raw data files.

    - `pipeline/dataset/raw/spy_YYYY_MM.csv` contains the raw intraday OHLCV data for YYYY-MM.

    - `pipeline/dataset/download.ipynb` contains the code for downloading the raw intraday OHLCV data using the Alpha Vantage free API.

    - `pipeline/dataset/eda.ipynb` contains the code for examining the processed data.

    - `pipeline/data.py` contains data-related classes and functions.

    - `pipeline/model.py` contains model-related classes and functions.

    - `pipeline/train.py` contains training-related functions.

    - `pipeline/utils.py` contains additional utility classes and functions.

- `tuning.py` contains the code for hyperparameter tuning.

- `main.py` contains the code for training the model.


## Installation

Install the dependencies with 

```
pip install -r requirements.txt
```