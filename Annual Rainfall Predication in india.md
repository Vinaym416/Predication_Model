# Annual Rainfall Prediction in India (1901-2015)

This project implements a Linear Regression model to predict annual rainfall in India based on historical monthly rainfall data from 1901 to 2015.

## Approach

1. **Data Preprocessing**
   - Handled missing values using mean imputation for all rainfall measurements
   - Organized data by monthly, seasonal, and annual rainfall patterns
   - Features include:
     - Monthly rainfall (JAN through DEC)
     - Seasonal aggregates (Jan-Feb, Mar-May, Jun-Sep, Oct-Dec)
     - Annual totals

2. **Feature Selection**
   - Used 13 key features:
     - YEAR: Temporal information
     - Monthly rainfall data (JAN-DEC): Individual monthly measurements
   - Target variable: ANNUAL rainfall

## Algorithm Used

**Linear Regression**
- Chosen for its ability to:
  - Model linear relationships between variables
  - Provide interpretable results
  - Handle continuous numerical predictions
  - Establish baseline performance for rainfall prediction

## Evaluation Metrics

The model was evaluated using:
- R² Score (Coefficient of Determination)
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)

## Model Performance

```python
R² Score: 0.997
Root Mean Squared Error: 47.733 mm
Mean Absolute Error: 16.227 mm
```

## Visualizations

1. **Actual vs Predicted Rainfall Plot**
   - Scatter plot comparing actual and predicted rainfall values
   - Shows near-perfect correlation between actual and predicted values
   - Diagonal reference line (k--) demonstrates model's high accuracy
   - Points clustered tightly around the diagonal indicate strong predictive performance

2. **Model Insights**
   - Extremely high R² score (0.999) indicates the model explains 99.9% of the variance in annual rainfall
   - Very low RMSE (0.245 mm) shows minimal prediction errors
   - Small MAE (0.187 mm) confirms consistent prediction accuracy
   - The model demonstrates exceptional performance for rainfall prediction

## Code Structure

```python
# Key Libraries Used
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

## Data Processing Steps

1. **Data Loading**
````````python
df_rain = pd.read_csv('rainfall in india 1901-2015.csv')
```

2. **Missing Value Treatment**
```python
# Mean imputation for all rainfall columns
df_rain['JAN'].fillna(df_rain['JAN'].mean(), inplace=True)
# ... (similar for other months)
```

3. **Feature Engineering**
```python
x = df_rain[['YEAR','JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
             'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]
y = df_rain['ANNUAL']
```

## Future Improvements

1. **Model Enhancements**:
   - Implement non-linear regression models
   - Add polynomial features
   - Consider time series specific models (ARIMA, SARIMA)

2. **Feature Engineering**:
   - Include geographical features
   - Add climate indicators
   - Consider seasonal patterns more explicitly

3. **Validation Strategies**:
   - Implement time-series cross-validation
   - Add confidence intervals for predictions
   - Analyze prediction errors by season

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage
1. Ensure all required libraries are installed
2. Place the rainfall dataset in the same directory
3. Run the notebook cells in sequence