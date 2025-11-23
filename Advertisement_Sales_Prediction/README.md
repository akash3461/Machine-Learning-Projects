# Marvellous Advertisement Sales Prediction (Jupyter Notebook Version)

## 🧠 Objective

To predict **sales revenue** based on advertisement expenditure using **Linear Regression** and visualize how different media platforms (**TV**, **Radio**, **Newspaper**) impact sales.

---

## 📦 Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from datetime import datetime
import joblib
from pathlib import Path
```

---

## 📂 Step 2: Load Dataset

```python
# Load the dataset
file_path = 'Advertising.csv'
df = pd.read_csv(file_path)

# Display first few rows
df.head()
```

---

## 🔍 Step 3: Explore the Data

```python
# Basic info
df.info()

# Summary statistics
df.describe()

# Check missing values
df.isnull().sum()
```

---

## 📈 Step 4: Visualize Relationships

```python
# Correlation heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot for relationships
sns.pairplot(df)
plt.show()
```

---

## ✂️ Step 5: Split the Data

```python
X = df[['TV', 'radio', 'newspaper']]
y = df[['sales']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🧮 Step 6: Build and Train the Model

```python
# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Train model
pipeline.fit(X_train, y_train)
```

---

## 📊 Step 7: Evaluate the Model

```python
y_pred = pipeline.predict(X_test)

mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
```

---

## 📉 Step 8: Visualize Predictions

```python
# Actual vs Predicted
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True)
plt.show()

# Residual plot
residuals = y_test.values.flatten() - y_pred.flatten()
plt.figure(figsize=(7,5))
plt.scatter(y_pred, residuals, color='red', alpha=0.6)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

---

## 📊 Step 9: Feature Importance

```python
regressor = pipeline.named_steps['regressor']
coefficients = regressor.coef_[0]
intercept = regressor.intercept_

importance = pd.Series(coefficients, index=X.columns)
importance.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Feature Importance')
plt.ylabel('Coefficient Value')
plt.show()

print('Model Coefficients:')
for col, coef in zip(X.columns, coefficients):
    print(f'{col}: {coef:.4f}')
print('Intercept:', intercept)
```

---

## 💾 Step 10: Save the Trained Model

```python
artifacts_dir = Path('artifacts/models')
artifacts_dir.mkdir(parents=True, exist_ok=True)

filename = artifacts_dir / f"marvellous_ad_sales_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
joblib.dump(pipeline, filename)
print(f"Model saved at: {filename}")
```

---

## 🧾 Step 11: Insights & Observations

```markdown
1. **TV advertising** shows the strongest correlation with sales.
2. **Radio** also impacts sales but to a lesser extent.
3. **Newspaper ads** have minimal effect on sales.
4. The model achieved an **R² Score of around 0.90**, showing strong prediction accuracy.
5. The residuals are evenly distributed, indicating a good model fit.
```

---

## 👨‍💻 Author

**Akash Chavan**
*Date: 9/11/2025*
