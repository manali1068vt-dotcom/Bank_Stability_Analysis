import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Bank_Stability_Analysis_Complete.csv')

# Calculate ratios
df['Profit_Margin'] = (df['Net_Income'] / df['Revenue']) * 100
df['Cost_Ratio'] = (df['Operating_Costs'] / df['Revenue']) * 100
df['D_E_Ratio'] = df['Debt'] / df['Equity']

# Normalize ratios (Min-Max Scaling)
scaler = MinMaxScaler()

df['Profit_Margin_Norm'] = scaler.fit_transform(df[['Profit_Margin']])
df['Cost_Ratio_Norm'] = 1 - scaler.fit_transform(df[['Cost_Ratio']])
df['D_E_Ratio_Norm'] = 1 - scaler.fit_transform(df[['D_E_Ratio']])

# Calculate Health Score (weighted composite)
df['Health_Score'] = (
    0.5 * df['Profit_Margin_Norm'] +
    0.4 * df['Cost_Ratio_Norm'] +
    0.1 * df['D_E_Ratio_Norm']
) * 100

# Classify risk categories
def classify_risk(score):
    if score > 65:
        return 'Low Risk / Healthy'
    elif score >= 40:
        return 'Medium Risk / Needs Watching'
    else:
        return 'High Risk / At-Risk'

df['Risk_Category'] = df['Health_Score'].apply(classify_risk)

# Descriptive statistics
desc_stats = df[['Revenue', 'Net_Income', 'Operating_Costs', 'Profit_Margin', 'Cost_Ratio', 'D_E_Ratio', 'Health_Score']].describe()

# Correlation analysis
correlation_matrix = df[['Profit_Margin', 'Cost_Ratio', 'D_E_Ratio', 'Health_Score']].corr()

# Regression analysis (predict 2025 Health Score)
results = []
for bank in df['Bank'].unique():
    bank_data = df[df['Bank'] == bank]
    X = bank_data['Year'].values.reshape(-1, 1)
    y = bank_data['Health_Score'].values
    model = LinearRegression().fit(X, y)
    pred_2025 = model.predict([[2025]])
    results.append({'Bank': bank, 'R2': model.score(X, y), 'Pred_2025': pred_2025[0]})
regression_results = pd.DataFrame(results)

# Output key results
print("Descriptive Statistics:")
print(desc_stats)
print("\nCorrelation Matrix:")
print(correlation_matrix)
print("\nRegression Predictions for 2025:")
print(regression_results)

# Visualizations
plt.figure(figsize=(12, 8))

# Bar chart: Health Scores by bank (2024)
plt.subplot(2, 3, 1)
sns.barplot(data=df[df['Year'] == 2024], x='Bank', y='Health_Score', hue='Risk_Category', dodge=False)
plt.title('2024 Health Scores by Bank')
plt.xticks(rotation=45)

# Boxplot: Health Score distribution by bank
plt.subplot(2, 3, 2)
sns.boxplot(data=df, x='Bank', y='Health_Score')
plt.title('Health Score Distribution by Bank')
plt.xticks(rotation=45)

# Line plot: Health Score trend 2019-2024
plt.subplot(2, 3, 3)
sns.lineplot(data=df, x='Year', y='Health_Score', hue='Bank', marker='o')
plt.title('Health Score Trend (2019-2024)')

# Histogram: Health Score distribution
plt.subplot(2, 3, 4)
sns.histplot(df['Health_Score'], bins=10, kde=True)
plt.title('Health Score Distribution')

# Correlation heatmap
plt.subplot(2, 3, 5)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')

# Bar chart: 2025 Predictions
plt.subplot(2, 3, 6)
sns.barplot(data=regression_results, x='Bank', y='Pred_2025')
plt.title('2025 Predicted Health Scores')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Save processed data
df.to_csv('Bank_Stability_Analysis_Results.csv', index=False)
