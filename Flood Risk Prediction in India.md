# Flood Risk Prediction in India

This project implements machine learning models to predict flood risks in different regions of India based on various environmental and geographical factors.

## Approach

1. **Data Preprocessing**
   - Handled missing values using mean imputation for numerical features and mode for categorical features
   - Encoded categorical variables (Land Cover and Soil Type) using numerical mapping
   - Balanced the dataset using SMOTE to address class imbalance

2. **Feature Engineering**
   - Selected 13 key features including:
     - Geographical: Latitude, Longitude, Elevation
     - Environmental: Rainfall, Temperature, Humidity
     - Hydrological: River Discharge, Water Level
     - Land characteristics: Land Cover, Soil Type
     - Demographic: Population Density, Infrastructure
     - Historical data: Past flood records

## Algorithms Used

1. **Random Forest Classifier**
   - Chosen for its ability to:
     - Handle non-linear relationships
     - Manage multiple features effectively
     - Provide feature importance rankings
     - Reduce overfitting through ensemble learning
   - Parameters:
     - n_estimators: 100
     - random_state: 42

2. **Logistic Regression** (comparison model)
   - Used as a baseline model
   - Demonstrates the advantage of Random Forest over linear models

## Evaluation Metrics

The models were evaluated using:
- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Model Performance

### Random Forest Results
- Overall Accuracy: 100%
- Flood Prediction:
  - Precision: 100%
  - Recall: 100%
  - F1-Score: 100%
- No Flood Prediction:
  - Precision: 100%
  - Recall: 100%
  - F1-Score: 100%

### Logistic Regression Results (Comparison)
- Overall Accuracy: 51%
- Flood Prediction:
  - Precision: 51%
  - Recall: 49%
  - F1-Score: 50%
- No Flood Prediction:
  - Precision: 51%
  - Recall: 53%
  - F1-Score: 52%

## Data Balancing with SMOTE

### Why SMOTE?
- The original dataset had an imbalanced distribution of flood/no-flood cases
- Imbalanced data can lead to biased models that perform poorly on minority class
- SMOTE (Synthetic Minority Over-sampling Technique) was used to create a balanced dataset

### SMOTE Implementation
```python
from imblearn.over_sampling import SMOTE

# Initialize SMOTE with random state for reproducibility
smote = SMOTE(random_state=42)

# Apply SMOTE to generate synthetic samples
X_resampled, y_resampled = smote.fit_resample(x, y)
```

### SMOTE Parameters
- random_state=42: Ensures reproducibility of results
- sampling_strategy='auto': Automatically determines the number of samples to generate
- k_neighbors=5 (default): Number of nearest neighbors to use for synthetic sample generation

### Impact of SMOTE
- Balanced class distribution in training data
- Improved model's ability to detect flood events
- Better generalization for minority class prediction
- Enhanced reliability of model performance metrics

## Graphs/Visualizations

1. **Confusion Matrix** 
   - Visual representation of model predictions vs actual values
   - Clear distinction between true positives/negatives and false positives/negatives
   - Generated using seaborn's heatmap with a 'Blues' colormap

2. **Performance Metrics Summary**
   - Bar charts comparing key metrics across models
   - Visual representation of precision, recall, and F1-scores

## Requirements
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
```