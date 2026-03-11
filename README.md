# Amazon_ml_challenge
!still working to improve
# Product Price Prediction using Multimodal ML

This repository contains our solution for the e-commerce product price prediction challenge, where the goal is to predict the optimal price of a product from its textual catalog content and product metadata.

The solution focuses primarily on **text-based understanding with transformer embeddings**, supported by **hand-engineered numerical features** and additional **ensemble experiments** to improve generalization.

---

## Problem Statement

In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction. The objective of this challenge is to build a machine learning model that analyzes product details and predicts the price of a product.

Product pricing is influenced by multiple factors such as:

- brand and product naming
- specifications and descriptive attributes
- product quantity and unit information
- overall packaging and content richness

The task is to model these relationships and generate accurate price predictions for unseen products.

---

## Dataset Description

The dataset contains the following columns:

- **sample_id**: Unique identifier for each sample
- **catalog_content**: Concatenated text field containing product title, product description, bullet points, and Item Pack Quantity (IPQ) related information
- **image_link**: Public URL for the product image
- **price**: Target variable, available only in the training set

### Dataset Size

- **Training set**: 75,000 samples
- **Test set**: 75,000 samples

### Required Submission Format

The final submission file must be a CSV with exactly 2 columns:

- `sample_id`
- `price`

Each test `sample_id` must have exactly one corresponding predicted price.

---

## Approach Overview

Our solution was built in multiple stages:

### 1. Transformer-based text regression
We use **DeBERTa-v3-base** as the primary backbone to learn semantic representations from product text.

Since `catalog_content` contains structured but noisy concatenated information, we first apply custom preprocessing to extract the most important signals:

- product name
- quantity value
- unit
- first few bullet points

This helps compress long catalog text into a more informative representation while keeping inference efficient.

### 2. Manual feature engineering
In addition to text embeddings, we extract a small set of structured numerical features, such as:

- quantity value
- unit type indicators
- text length
- bullet point count

These features are concatenated with the transformer representation before regression.

### 3. Target transformation
Because price distributions are typically skewed, we apply:

- `log1p` transformation
- `RobustScaler`

This improves optimization stability and reduces sensitivity to outliers.

### 4. Validation-driven training
We train using a **60k / 15k train-validation split** and track **SMAPE** on the validation set to select the best checkpoint.

### 5. Ensemble experiments
Beyond the main transformer model, we also experimented with:

- TF-IDF features
- RandomForest-based residual correction
- XGBoost regression on engineered features
- weighted blending between DeBERTa and XGBoost outputs

These experiments were used to explore whether lightweight structured models could complement transformer predictions.

---

## Model Architecture

### Primary Model: `FastDebertaPredictor`

The main model consists of:

- **DeBERTa-v3-base** text encoder
- partial fine-tuning of the last 4 transformer layers
- concatenation of transformer CLS embedding with engineered features
- multi-layer regression head

### Regression Head

- Linear(768 + engineered_features → 384)
- ReLU + Dropout
- Linear(384 → 192)
- ReLU + Dropout
- Linear(192 → 96)
- ReLU
- Linear(96 → 1)

### Training Details

- **Loss**: Huber Loss
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Weight Decay**: 0.01
- **Scheduler**: Linear warmup + decay
- **Batch Size**: 32
- **Epochs**: planned 30
- **Gradient Clipping**: 1.0

---

## Text Preprocessing Strategy

The raw `catalog_content` field may contain multiple sections such as:

- Item Name
- Value
- Unit
- Bullet Points
- Product Description

Instead of feeding the entire raw text directly, we extract the most informative parts.

Example preprocessing output:

- `PRODUCT: <item name>`
- `SIZE: <value> <unit>`
- `FEAT: <bullet 1>`
- `FEAT: <bullet 2>`
- `FEAT: <bullet 3>`

This improves signal density and helps the transformer focus on the strongest pricing cues.

---

## Feature Engineering

The structured features currently used in the main model are:

- extracted numeric value
- binary flags for important unit families:
  - ounce / oz
  - count / ct / piece
  - fluid / fl
- total text length
- number of bullet points

Additional ensemble experiments included:

- more detailed unit encodings
- token count
- uppercase count
- content presence indicators
- TF-IDF + SVD reduced text features

---

## Validation Metric

We evaluate the model using **SMAPE** (Symmetric Mean Absolute Percentage Error):

\[
SMAPE = 100 \times mean\left(\frac{2 \cdot |pred - actual|}{|pred| + |actual| + \epsilon}\right)
\]

This metric is suitable for price prediction because it normalizes absolute error relative to the scale of the target.

---

## Training Results

The main DeBERTa-based model showed steady improvement over training.

### Best validation performance observed
- **Best Validation SMAPE:** `47.6725%`
- **Best Epoch:** `18`

This was the strongest result obtained during the logged training run before interruption.

---

## Repository Structure

```text
.
├── train.csv
├── test.csv
├── best_model_weights.pt
├── feature_scaler.pkl
├── target_scaler.pkl
├── submission.csv
├── improved_submission.csv
├── ensemble_submission.csv
└── README.md
