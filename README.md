# Blackjack

This project documents a three-step Blackjack notebook workflow: baseline training, count-aware training, and a direct comparison.

## Setup

```bash
git clone https://github.com/Tom-N-M/ml-blackjack.git
cd ml-blackjack
conda env create -f environment.yml
conda activate ml-blackjack
python -m ipykernel install --user --name=ml-blackjack --display-name "Python (ml-blackjack)"
jupyter lab
```

## Project Structure

```text
ml-blackjack/
├── blackjack_env.py          # Custom Blackjack Gymnasium environment with count features
├── environment.yml           # Conda environment definition
├── notebooks/
│   ├── 01_baseline.ipynb     # Baseline agent without card-count history
│   ├── 02_counting.ipynb     # Counting agent built on blackjack_env.py
│   └── 03_comparison.ipynb   # Baseline vs counting comparison
├── models/                   # Pickled agent artifacts written by notebooks 01 and 02
└── README.md                 # Project overview and setup
```

Run the notebooks in order. Notebook 03 expects the saved artifacts from notebooks 01 and 02.