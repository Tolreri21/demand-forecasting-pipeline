# Demand Forecasting Pipeline

End-to-end project for demand forecasting using tabular retail data. The pipeline covers
data preprocessing, feature preparation, and model training with scikit-learn and PyTorch.

## Project layout
- `src/preprocess.py`: data loading, feature creation, and train/test split export.
- `src/train.py`: training utilities and modeling setup (scikit-learn + PyTorch).
- `data/`: raw and processed datasets.
- `models/`: saved model artifacts.
- `notebooks/`: experiments and analysis.

## Getting started
1) Create a virtual environment and install dependencies:
   - `pip install -r requirements.txt`
2) Run preprocessing to generate training data.
3) Run training to fit a forecasting model.

## Notes
- This project is intended for local experimentation; adjust paths as needed.
