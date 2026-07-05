# Disaster Tweet Classification

This project classifies disaster-related tweets as **relevant** or **not relevant** using classical machine learning, deep learning, transformer-based, and hybrid neural architectures.

The main notebook is:

```text
disaster tweet classification.ipynb
```

## Project Overview

During disaster events, social media posts can contain important information for emergency response, situational awareness, and humanitarian decision-making. This notebook builds and evaluates multiple text classification models to identify whether a tweet is disaster-relevant.

The project compares model performance across both in-domain and cross-domain disaster datasets.

## Datasets

The notebook uses multiple disaster-event datasets:

- `2015_Nepal_Earthquake_train.tsv`
- `2015_Nepal_Earthquake_dev.tsv`
- `2015_Nepal_Earthquake_test.tsv`
- `2013_Queensland_Floods_train.tsv`
- `2013_Queensland_Floods_dev.tsv`
- `2013_Queensland_Floods_test.tsv`
- `2014_Chile_Earthquake_en.csv`

The labels are mapped into binary classes:

| Label | Meaning |
|---|---|
| `1` | Relevant disaster tweet |
| `0` | Not relevant disaster tweet |

For the Chile earthquake dataset, selected categories such as personal updates, sympathy/support, and money donations are mapped to the non-relevant class, while the remaining categories are treated as relevant.

## Experiments

The notebook evaluates models under four experiment settings:

| Experiment | Setting |
|---|---|
| `EXP1` | Master training data to master test data |
| `EXP2` | Master training data to Chile earthquake test data |
| `EXP3` | Master training data to Nepal earthquake test data |
| `EXP4` | Master training data to Queensland floods test data |

These experiments help compare in-domain performance and generalization under domain shift.

## Workflow

The notebook follows this pipeline:

1. Load Nepal earthquake, Queensland floods, and Chile earthquake datasets.
2. Map labels into binary classes.
3. Merge train/dev data into a master training set.
4. Clean and preprocess tweet text.
5. Perform exploratory data analysis.
6. Train and evaluate classical machine learning models.
7. Train and evaluate deep learning models.
8. Fine-tune transformer models.
9. Train hybrid Transformer-RNN models.
10. Compare results using plots and tabular summaries.

## Text Preprocessing

The preprocessing pipeline includes:

- Lowercasing text.
- Removing or normalizing URLs.
- Replacing user mentions.
- Handling hashtags.
- Removing repeated characters.
- Removing unnecessary symbols.
- Tokenizing tweets.
- Preparing cleaned text for TF-IDF, neural, and transformer models.

## Models Implemented

### Classical Machine Learning

- TF-IDF + Linear SVM
- TF-IDF + Logistic Regression

### Classical Deep Learning

- TextCNN with GloVe embeddings
- BiLSTM with Attention
- BiLSTM with GloVe embeddings and Attention

### Transformer Models

- BERT-based sequence classification

### Hybrid Models

- BERT + BiLSTM + Attention

## Evaluation Metrics

The notebook evaluates each model using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve
- Precision-recall curve

All model results are stored in a common `RESULTS` list and converted into summary tables for comparison.

## Result Visualization

The notebook generates:

- Dataset size plots before and after cleaning.
- Individual model-family comparison plots.
- Best-model comparison plots.
- Overall model comparison charts.
- Tabular metric summaries for each model family.

<!-- ## Project Structure

```text
.
├── disaster tweet classification.ipynb
├── 2013_Queensland_Floods_train.tsv
├── 2013_Queensland_Floods_dev.tsv
├── 2013_Queensland_Floods_test.tsv
├── 2015_Nepal_Earthquake_train.tsv
├── 2015_Nepal_Earthquake_dev.tsv
├── 2015_Nepal_Earthquake_test.tsv
├── 2014_Chile_Earthquake_en.csv
├── chile_test.csv
└── README.md
``` -->

## Requirements

Install the following Python packages before running the notebook:

```bash
pip install numpy pandas matplotlib scikit-learn wordcloud torch transformers tqdm
```

Optional, if running inside Jupyter:

```bash
pip install notebook ipykernel
```

## How to Run

1. Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

2. Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn wordcloud torch transformers tqdm notebook ipykernel
```

3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open and run:

```text
disaster tweet classification.ipynb
```

Run the notebook cells from top to bottom so that datasets, preprocessing functions, model definitions, training loops, and result visualizations execute in the correct order.

## Reproducibility

The notebook sets a fixed random seed:

```python
SEED = 42
```

This helps make data splitting, model initialization, and evaluation more reproducible.

## Notes

- Transformer and hybrid models require more compute than classical ML models.
- GPU acceleration is recommended for BERT and hybrid deep learning experiments.
- Dataset files should be placed in the same directory as the notebook unless paths are updated manually.
- The generated `chile_test.csv` file is created from the Chile earthquake CSV preprocessing step.

## Author

Roushan

