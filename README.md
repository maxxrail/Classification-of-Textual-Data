# Classification-of-Textual-Data

## Overview
This Jupyter notebook implements a series of machine learning experiments on text datasets, including the IMDB movie reviews dataset and the 20 Newsgroups dataset. It covers:

- Data pre‑processing and feature extraction (Bag‑of‑Words, TF‑IDF)
- Custom implementations of multiclass softmax regression and linear regression
- Feature selection using mutual information
- Classification experiments using linear regression, logistic regression, k‑NN, and decision trees
- Regression experiments and analysis
- Evaluation metrics and plotting functions for visualizing top features and heatmaps

## Requirements
- Python 3.7 or higher
- pandas
- numpy
- scikit‑learn
- matplotlib
- seaborn
- scipy
- tqdm

Install dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy tqdm
```

## Data Setup
Place the ACL IMDB dataset files in the following relative paths:

```
../aclImdb/imdb.vocab
../aclImdb/train/labeledBow.feat
../aclImdb/test/labeledBow.feat
```

The 20 Newsgroups dataset is automatically fetched by scikit‑learn.

## Notebook Structure
1. **Pre‑Process Data**
   Import libraries and define helper classes (`MultiClassRegression`, `LinearRegression`).

2. **Extract, Transform, Load (ETL)**
   Functions for loading vocabulary, parsing review features, and preparing feature matrices.

3. **Evaluation and Plotting Functions**
   Functions to evaluate multiclass classification and plot top features.

4. **Experiments**
   Nine experiments (`experiment_one()` through `experiment_nine()`) demonstrating various ML techniques.

5. **Results**
   Each experiment outputs plots saved as PNG files (e.g. `experiment_9_heatmap_results.png`).

## Usage
1. Open `assignment2_group-76.ipynb` in Jupyter Notebook.
2. Run all cells in order.
3. Review generated plots and output to analyze model performance and extracted features.

## Results Summary
- **Experiment 1**: Top features for linear vs logistic regression.
- **Experiment 2**: Binary vs multiclass classification on the 20 Newsgroups dataset.
- **Experiment 3**: k‑NN classification performance.
- **Experiment 4**: Decision tree classification analysis.
- **Experiment 5**: Linear regression on continuous targets.
- **Experiment 6**: Advanced regression techniques.
- **Experiment 7**: [Add description here]
- **Experiment 8**: [Add description here]
- **Experiment 9**: Heatmap of top features for each class in the 20 Newsgroups dataset (`experiment_9_heatmap_results.png`).

## Author
Group 76

