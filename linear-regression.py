import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Step 1: Data Preparation
# Create a DataFrame with the data
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November']

data = {
    'Month': months + months,
    'Year': [2023] * 11 + [2024] * 11,
    'Gasoline_Price': [68.7375, 67.9272, 61.924, 64.16598, 58.5165, 60.3612, 62.6088, 68.5152, 68.7159, 63.6048, 63.0653,
                      63.2461, 67.272, 65.3328, 62.0755, 62.9584, 63.983, 63.7432, 65.7685, 56.07, 56.154, 55.7555],
    'Crude_Oil_Price': [27.9780177, 28.24329411, 28.06403039, 29.04127246, 26.21276834, 28.00353111, 28.03567084, 
                       30.54082158, 28.64021712, 31.90144414, 29.15696629,
                       27.79366552, 28.54357274, 28.07330914, 32.01281551, 30.42275406, 30.84032657, 30.66588841,
                       27.89229685, 29.14115053, 27.04132413, 26.75596873]
}

df = pd.DataFrame(data)

# Create a time series index
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%B')
df = df.sort_values('Date')
df.reset_index(drop=True, inplace=True)

print("Step 1: Data Preparation Complete")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# Step 2: Exploratory Data Analysis (EDA)
print("\nStep 2: Exploratory Data Analysis")

# Create a function to save figures
def save_figure(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Plot the time series of both variables
fig, ax = plt.subplots(2, 1, figsize=(14, 10))

# Plot Gasoline Price
sns.lineplot(x='Date', y='Gasoline_Price', data=df, marker='o', color='blue', ax=ax[0])
ax[0].set_title('Gasoline Price Trend (2023-2024)', fontsize=14)
ax[0].set_ylabel('Gasoline Price (PHP/liter)')
ax[0].grid(True, alpha=0.3)

# Plot Crude Oil Price
sns.lineplot(x='Date', y='Crude_Oil_Price', data=df, marker='o', color='green', ax=ax[1])
ax[1].set_title('Crude Oil Price Trend (2023-2024)', fontsize=14)
ax[1].set_ylabel('Crude Oil Price (PHP/liter)')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, 'price_trends.png')

# Scatter plot to visualize the relationship
fig = plt.figure(figsize=(10, 6))
sns.scatterplot(x='Crude_Oil_Price', y='Gasoline_Price', data=df, hue='Year', palette=['blue', 'red'], s=100)
plt.title('Relationship between Crude Oil Price and Gasoline Price', fontsize=14)
plt.xlabel('Crude Oil Price (PHP/liter)')
plt.ylabel('Gasoline Price (PHP/liter)')
plt.grid(True, alpha=0.3)
save_figure(fig, 'price_relationship.png')

# Calculate correlation
correlation = df['Crude_Oil_Price'].corr(df['Gasoline_Price'])
print(f"Correlation between Crude Oil Price and Gasoline Price: {correlation:.4f}")

# Create a heatmap of correlations
fig = plt.figure(figsize=(8, 6))
correlation_matrix = df[['Crude_Oil_Price', 'Gasoline_Price']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.4f')
plt.title('Correlation Matrix', fontsize=14)
save_figure(fig, 'correlation_heatmap.png')

# Step 3: Feature Engineering
print("\nStep 3: Feature Engineering")

# Add lag features (previous month's crude oil price)
df['Crude_Oil_Price_Lag1'] = df['Crude_Oil_Price'].shift(1)
df['Crude_Oil_Price_Lag2'] = df['Crude_Oil_Price'].shift(2)

# Calculate price percentage changes
df['Crude_Oil_Price_pct_change'] = df['Crude_Oil_Price'].pct_change()
df['Gasoline_Price_pct_change'] = df['Gasoline_Price'].pct_change()

# Add month as a categorical feature for seasonality
df['Month_Num'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

# Create dummy variables for Month
df_with_dummies = pd.get_dummies(df, columns=['Month_Num'], prefix='Month', drop_first=True)

# Drop NA values from lag creation
df_clean = df.dropna().copy()
df_with_dummies_clean = df_with_dummies.dropna().copy()

print("Feature engineering complete. New columns added:")
new_columns = set(df.columns) - set(['Month', 'Year', 'Gasoline_Price', 'Crude_Oil_Price', 'Date'])
print(list(new_columns))

# Step 4: Basic Linear Regression Model
print("\nStep 4: Basic Linear Regression Model")

# Define features and target for the simplest model
X = df_clean[['Crude_Oil_Price']]
y = df_clean['Gasoline_Price']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel Coefficients:")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficient for Crude Oil Price: {model.coef_[0]:.4f}")
print(f"\nA 1 PHP increase in crude oil price is associated with a {model.coef_[0]:.4f} PHP increase in gasoline price")

print(f"\nModel Performance Metrics:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Step 5: Advanced Analysis with Statsmodels
print("\nStep 5: Advanced Analysis with Statsmodels")

# Add constant for statsmodels
X_sm = sm.add_constant(X)

# Fit the model
model_sm = sm.OLS(y, X_sm).fit()

# Print summary
print("\nStatsmodels Summary:")
print(model_sm.summary().tables[1])  # Print just the coefficients table

# Step 6: Model Validation and Cross-Validation
print("\nStep 6: Model Validation and Cross-Validation")

# Set up k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring='neg_mean_squared_error')

# Convert negative MSE to RMSE
rmse_scores = np.sqrt(-cv_scores)

print(f"Cross-validation RMSE scores: {rmse_scores}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
print(f"Standard deviation of RMSE: {np.std(rmse_scores):.4f}")

# Step 7: Results Visualization and Model Evaluation
print("\nStep 7: Results Visualization and Model Evaluation")

# Create a DataFrame for actual vs predicted values
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# Plot actual vs predicted
fig = plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.7)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Crude Oil Price (PHP/liter)')
plt.ylabel('Gasoline Price (PHP/liter)')
plt.title('Linear Regression: Actual vs Predicted Gasoline Prices', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
save_figure(fig, 'regression_results.png')

# Residual plot
fig = plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot', fontsize=14)
plt.grid(True, alpha=0.3)
save_figure(fig, 'residual_plot.png')

# Histogram of residuals
fig = plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals', fontsize=14)
plt.xlabel('Residual Value')
save_figure(fig, 'residual_histogram.png')

# Step 8: Multiple Model Comparison
print("\nStep 8: Multiple Model Comparison")

# Setup more complex features
X_multi = df_with_dummies_clean[['Crude_Oil_Price', 'Crude_Oil_Price_Lag1']]
y_multi = df_with_dummies_clean['Gasoline_Price']

X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_multi_train, y_multi_train)
    y_multi_pred = model.predict(X_multi_test)
    results[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_multi_test, y_multi_pred)),
        'R2': r2_score(y_multi_test, y_multi_pred),
        'MAE': mean_absolute_error(y_multi_test, y_multi_pred)
    }

results_df_models = pd.DataFrame(results).T
print(results_df_models)

# Create a bar chart to compare model performance
fig = plt.figure(figsize=(12, 6))
results_df_models['RMSE'].plot(kind='bar', color='skyblue')
plt.title('RMSE Comparison Across Models', fontsize=14)
plt.ylabel('RMSE (PHP)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
save_figure(fig, 'model_comparison_rmse.png')

fig = plt.figure(figsize=(12, 6))
results_df_models['R2'].plot(kind='bar', color='lightgreen')
plt.title('R² Comparison Across Models', fontsize=14)
plt.ylabel('R²')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
save_figure(fig, 'model_comparison_r2.png')

# Step 9: Hyperparameter Tuning
print("\nStep 9: Hyperparameter Tuning")

# For Ridge regression with GridSearchCV
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
}

# Create the GridSearchCV object
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search
grid_search.fit(X_multi_train, y_multi_train)

# Best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

# Use best model for predictions
best_model = grid_search.best_estimator_
best_pred = best_model.predict(X_multi_test)
best_rmse = np.sqrt(mean_squared_error(y_multi_test, best_pred))
best_r2 = r2_score(y_multi_test, best_pred)

print(f"Best Model Test RMSE: {best_rmse:.4f}")
print(f"Best Model Test R2: {best_r2:.4f}")

# Step 10: Price Elasticity Analysis
print("\nStep 10: Price Elasticity Analysis")

# Remove NaN and infinite values
elasticity_df = df_clean.dropna()
elasticity_df = elasticity_df[(elasticity_df['Crude_Oil_Price_pct_change'] != 0) & 
                            (np.isfinite(elasticity_df['Crude_Oil_Price_pct_change'])) &
                            (np.isfinite(elasticity_df['Gasoline_Price_pct_change']))]

# Calculate elasticity for each month
elasticity_df['elasticity'] = elasticity_df['Gasoline_Price_pct_change'] / elasticity_df['Crude_Oil_Price_pct_change']

# Calculate average elasticity
avg_elasticity = elasticity_df['elasticity'].mean()
print(f"Average Price Elasticity: {avg_elasticity:.4f}")
print("Note: Value > 1 means gasoline prices change proportionally more than crude oil prices")
print("      Value < 1 means gasoline prices change proportionally less than crude oil prices")

# Create a scatter plot for elasticity analysis
fig = plt.figure(figsize=(10, 6))
plt.scatter(elasticity_df['Crude_Oil_Price_pct_change'], elasticity_df['Gasoline_Price_pct_change'], alpha=0.7)
plt.xlabel('Crude Oil Price % Change')
plt.ylabel('Gasoline Price % Change')
plt.title('Price Elasticity Analysis', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
save_figure(fig, 'elasticity_analysis.png')

# Step 11: Time Series Analysis
print("\nStep 11: Time Series Analysis")

# Set the Date as index for time series analysis
ts_df = df.set_index('Date')

# Plot Monthly Average Prices
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111)

# Plot both series on the same axis using dual y-axis
color1, color2 = 'blue', 'green'

ax.set_xlabel('Date')
ax.set_ylabel('Gasoline Price (PHP/liter)', color=color1)
ax.plot(ts_df.index, ts_df['Gasoline_Price'], color=color1, marker='o')
ax.tick_params(axis='y', labelcolor=color1)

ax2 = ax.twinx()
ax2.set_ylabel('Crude Oil Price (PHP/liter)', color=color2)
ax2.plot(ts_df.index, ts_df['Crude_Oil_Price'], color=color2, marker='s')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Monthly Average Gasoline and Crude Oil Prices (2023-2024)', fontsize=14)
plt.grid(True, alpha=0.3)
save_figure(fig, 'monthly_average_prices.png')

# Try to decompose the gasoline price time series if enough data points
if len(ts_df) >= 24:  # Need at least 2x seasonal periods for seasonal_decompose
    try:
        decomposition = seasonal_decompose(ts_df['Gasoline_Price'], model='additive', period=12)
        
        fig = plt.figure(figsize=(14, 10))
        plt.subplot(411)
        plt.plot(decomposition.observed)
        plt.title('Observed', fontsize=12)
        plt.subplot(412)
        plt.plot(decomposition.trend)
        plt.title('Trend', fontsize=12)
        plt.subplot(413)
        plt.plot(decomposition.seasonal)
        plt.title('Seasonal', fontsize=12)
        plt.subplot(414)
        plt.plot(decomposition.resid)
        plt.title('Residual', fontsize=12)
        plt.tight_layout()
        save_figure(fig, 'time_series_decomposition.png')
    except:
        print("Not enough data points for seasonal decomposition.")
else:
    print("Not enough data points for seasonal decomposition.")

# Step 12: Interactive Visualization with Plotly
print("\nStep 12: Creating Interactive Visualization")

# Create an interactive time series plot
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                   subplot_titles=('Gasoline Price', 'Crude Oil Price'))

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['Gasoline_Price'], name='Gasoline Price', mode='lines+markers'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['Crude_Oil_Price'], name='Crude Oil Price', mode='lines+markers'),
    row=2, col=1
)

fig.update_layout(height=600, width=900, title_text="Interactive Price Trends")
fig.write_html("interactive_trends.html")

# Create an interactive scatter plot
fig = px.scatter(df, x='Crude_Oil_Price', y='Gasoline_Price', color='Year',
                trendline='ols', trendline_color_override='red',
                title='Interactive Relationship between Crude Oil and Gasoline Prices')
fig.write_html("interactive_scatter.html")

# Step 13: Final Summary
print("\nStep 13: Final Summary of Analysis")

print("\n=== FINAL SUMMARY ===")
print(f"1. Correlation between Crude Oil Price and Gasoline Price: {correlation:.4f}")
print(f"2. Basic Linear Regression Model R²: {r2:.4f}")
print(f"3. Linear Regression Model Equation: Gasoline Price = {model.intercept_:.4f} + {model.coef_[0]:.4f} × Crude Oil Price")
print(f"4. Best Performing Model: {results_df_models['R2'].idxmax()} with R² of {results_df_models['R2'].max():.4f}")
print(f"5. Average Price Elasticity: {avg_elasticity:.4f}")

print("\nAnalysis Complete! All visualizations have been saved to your working directory.")