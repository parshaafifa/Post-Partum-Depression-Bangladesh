# Postpartum Depression Factor Analysis and SEM

## Overview
This project analyzes postpartum depression (PPD) data from mothers in Bangladesh. It applies **factor analysis** and **structural equation modeling (SEM)** to explore the relationships between psychological stress, social support, bonding, rest, and EPDS scores.

## Features
- Preprocessing and encoding of categorical psychological and social variables
- Bartlett's test and KMO test for factorability
- Scree plot and eigenvalue analysis to determine optimal factors
- Factor analysis with Varimax rotation
- Heatmap visualization of factor loadings
- Extraction of factor scores for downstream analysis
- SEM modeling of relationships between social support, bonding, rest, psychological stress, and EPDS score

## Dataset
- `PPD_dataset_v2.csv` – contains survey responses of postpartum mothers
- Ensure the dataset is in the same folder as the Python script

## Usage
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn factor-analyzer semopy

# Run the script
python "factor_analysis.py"
