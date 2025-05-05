import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("coolwarm")
plt.rcParams['figure.figsize'] = (12, 8)

# Create DataFrame from provided data
months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']

gas_2023 = [68.7375, 67.9272, 61.924, 64.16598, 58.5165, 60.3612, 
            62.6088, 68.5152, 68.7159, 63.6048, 63.0653, 61.149]
gas_2024 = [63.2461, 67.272, 65.3328, 62.0755, 62.9584, 63.983, 
            63.7432, 65.7685, 56.07, 56.154, 55.7555, 57.8655]

oil_2023 = [27.9780177, 28.24329411, 28.06403039, 29.04127246, 26.21276834, 28.00353111,
            28.03567084, 30.54082158, 28.64021712, 31.90144414, 29.15696629, 28.83573688]
oil_2024 = [27.79366552, 28.54357274, 28.07330914, 32.01281551, 30.42275406, 30.84032657,
            30.66588841, 27.89229685, 29.14115053, 27.04132413, 26.75596873, 29.46635574]

# Create the dataset with date index
data = []
for i, month in enumerate(months):
    # 2023 data
    data.append({
        'Date': pd.to_datetime(f'2023-{i+1}-01'),
        'Month': month,
        'Year': 2023,
        'Gas_Price': gas_2023[i],
        'Oil_Price': oil_2023[i]
    })
    # 2024 data
    data.append({
        'Date': pd.to_datetime(f'2024-{i+1}-01'),
        'Month': month,
        'Year': 2024,
        'Gas_Price': gas_2024[i],
        'Oil_Price': oil_2024[i]
    })

df = pd.DataFrame(data)
df = df.sort_values('Date').reset_index(drop=True)

print("Dataset Preview:")
print(df.head())

# Data summary
print("\nData Summary Statistics:")
print(df.describe())

# Add derived features
df['Month_Num'] = df['Date'].dt.month
df['Price_Diff'] = df['Gas_Price'] - df['Oil_Price']
df['Price_Ratio'] = df['Gas_Price'] / df['Oil_Price']

# Create lagged features
for i in range(1, 4):
    df[f'Oil_Price_Lag_{i}'] = df['Oil_Price'].shift(i)

# Drop rows with NaN values created by lag features
df_with_lags = df.dropna().reset_index(drop=True)

# ============================================================================
# 6. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n\n================= 6. EXPLORATORY DATA ANALYSIS =================")

# EDA 1: Time Series Plot
plt.figure(figsize=(14, 8))
fig, ax1 = plt.subplots(figsize=(14, 8))
ax1.set_xlabel('Date')
ax1.set_ylabel('Gasoline Price (PHP/L)', color='tab:blue')
ax1.plot(df['Date'], df['Gas_Price'], marker='o', color='tab:blue', label='Gasoline Price')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Crude Oil Price (PHP/L)', color='tab:red')
ax2.plot(df['Date'], df['Oil_Price'], marker='s', color='tab:red', label='Crude Oil Price')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Add vertical line between years
plt.axvline(x=pd.to_datetime('2024-01-01'), color='gray', linestyle='--')
plt.title('Time Series of Gasoline and Crude Oil Prices in the Philippines (2023-2024)', fontsize=14)

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('time_series_plot.png', dpi=300)
plt.close()

print("1. Time Series Plot: Created visualization showing both prices over time")

# EDA 2: Scatter Plot with Regression Line
plt.figure(figsize=(12, 8))
sns.regplot(x='Oil_Price', y='Gas_Price', data=df, scatter_kws={'alpha':0.7, 's':100}, 
            line_kws={'color':'red'})
plt.title('Scatter Plot: Crude Oil Price vs Gasoline Price with Regression Line', fontsize=14)
plt.xlabel('Crude Oil Price (PHP/L)')
plt.ylabel('Gasoline Price (PHP/L)')

# Enhance the plot by coloring points by year
for year, color in zip([2023, 2024], ['blue', 'green']):
    year_data = df[df['Year'] == year]
    plt.scatter(year_data['Oil_Price'], year_data['Gas_Price'], 
                alpha=0.7, s=120, label=f'Year {year}')

plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('scatter_plot.png', dpi=300)
plt.close()

print("2. Scatter Plot: Created with regression line showing price relationship")

# EDA 3: Correlation Analysis
correlation = df[['Gas_Price', 'Oil_Price']].corr()
print("\n3. Correlation Analysis:")
print(correlation)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix: Gasoline and Crude Oil Prices', fontsize=14)
plt.savefig('correlation_heatmap.png', dpi=300)
plt.close()

# EDA 4: Seasonal Decomposition
# For Gas Price
try:
    gas_decomposition = seasonal_decompose(df['Gas_Price'], model='additive', period=12)
    plt.figure(figsize=(14, 10))
    plt.subplot(411)
    plt.plot(df['Date'], gas_decomposition.observed)
    plt.title('Gasoline Price - Observed', fontsize=12)
    plt.subplot(412)
    plt.plot(df['Date'], gas_decomposition.trend)
    plt.title('Gasoline Price - Trend Component', fontsize=12)
    plt.subplot(413)
    plt.plot(df['Date'], gas_decomposition.seasonal)
    plt.title('Gasoline Price - Seasonal Component', fontsize=12)
    plt.subplot(414)
    plt.plot(df['Date'], gas_decomposition.resid)
    plt.title('Gasoline Price - Residual Component', fontsize=12)
    plt.tight_layout()
    plt.savefig('gas_seasonal_decomposition.png', dpi=300)
    plt.close()
    
    print("\n4. Seasonal Decomposition: Performed for both price series")
except:
    print("\n4. Seasonal Decomposition: Could not perform due to insufficient data")

# EDA 5: Box Plots by Year
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Year', y='Gas_Price', data=df)
plt.title('Gasoline Price Distribution by Year', fontsize=12)
plt.subplot(1, 2, 2)
sns.boxplot(x='Year', y='Oil_Price', data=df)
plt.title('Crude Oil Price Distribution by Year', fontsize=12)
plt.tight_layout()
plt.savefig('box_plots_by_year.png', dpi=300)
plt.close()

print("5. Box Plots: Created to compare distributions between 2023 and 2024")

# EDA 6: Lag Analysis
plt.figure(figsize=(10, 8))
for i, lag in enumerate(['Oil_Price', 'Oil_Price_Lag_1', 'Oil_Price_Lag_2', 'Oil_Price_Lag_3']):
    if lag == 'Oil_Price':
        label = 'Current Month'
    else:
        label = f'{lag[-1]} Month(s) Lag'
    
    if lag in df_with_lags.columns:
        plt.subplot(2, 2, i+1)
        sns.regplot(x=lag, y='Gas_Price', data=df_with_lags, scatter_kws={'alpha':0.6}, 
                    line_kws={'color':'red'})
        plt.title(f'Gas Price vs Oil Price ({label})', fontsize=10)
        
        # Calculate and display correlation
        correlation = df_with_lags[['Gas_Price', lag]].corr().iloc[0,1]
        plt.annotate(f'Correlation: {correlation:.3f}', 
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.savefig('lag_analysis.png', dpi=300)
plt.close()

print("6. Lag Analysis: Visualized how gasoline prices respond to crude oil price changes")

# ============================================================================
# 7. MODEL SELECTION
# ============================================================================

print("\n\n================= 7. MODEL SELECTION =================")

# Prepare data for modeling
X_simple = df[['Oil_Price']]
y = df['Gas_Price']

# Model 1: Simple Linear Regression
model_simple = LinearRegression()
model_simple.fit(X_simple, y)

# Print simple model results
print("\nModel 1: Simple Linear Regression")
print(f"Coefficient: {model_simple.coef_[0]:.4f}")
print(f"Intercept: {model_simple.intercept_:.4f}")
print(f"R² Score: {model_simple.score(X_simple, y):.4f}")

# For more detailed statistics
X_simple_sm = sm.add_constant(X_simple)
model_simple_sm = sm.OLS(y, X_simple_sm).fit()
print(model_simple_sm.summary().tables[1])

# Model 2: Multiple Linear Regression with Lagged Variables
X_multi = df_with_lags[['Oil_Price', 'Oil_Price_Lag_1', 'Oil_Price_Lag_2', 'Oil_Price_Lag_3']]
y_multi = df_with_lags['Gas_Price']

# Add month dummies for seasonality
month_dummies = pd.get_dummies(df_with_lags['Month_Num'], prefix='Month', drop_first=True)
X_multi_seasonal = pd.concat([X_multi, month_dummies], axis=1)

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

model_multi_seasonal = LinearRegression()
model_multi_seasonal.fit(X_multi_seasonal, y_multi)

# Print multiple model results
print("\nModel 2: Multiple Linear Regression with Lagged Variables")
print(f"R² Score (without seasonality): {model_multi.score(X_multi, y_multi):.4f}")
print(f"R² Score (with seasonality): {model_multi_seasonal.score(X_multi_seasonal, y_multi):.4f}")

# Model 3: Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_simple)

model_poly = LinearRegression()
model_poly.fit(X_poly, y)

# Print polynomial model results
print("\nModel 3: Polynomial Regression (Degree 2)")
print(f"R² Score: {model_poly.score(X_poly, y):.4f}")

# ============================================================================
# 8. MODEL TRAINING & TUNING
# ============================================================================

print("\n\n================= 8. MODEL TRAINING & TUNING =================")

# Split data into training and testing sets (18 months training, 6 months testing)
train_size = 18
test_size = len(df) - train_size

df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

X_train = df_train[['Oil_Price']]
y_train = df_train['Gas_Price']
X_test = df_test[['Oil_Price']]
y_test = df_test['Gas_Price']

# Train the model on training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Performance Metrics on Test Set:")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=3)
cv_scores = []

for train_index, test_index in tscv.split(X_simple):
    X_cv_train, X_cv_test = X_simple.iloc[train_index], X_simple.iloc[test_index]
    y_cv_train, y_cv_test = y.iloc[train_index], y.iloc[test_index]
    
    model_cv = LinearRegression()
    model_cv.fit(X_cv_train, y_cv_train)
    
    score = model_cv.score(X_cv_test, y_cv_test)
    cv_scores.append(score)

print(f"\nTime Series Cross-Validation R² Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Average CV R² Score: {np.mean(cv_scores):.4f}")

# ============================================================================
# 9. RESULTS AND EVALUATION
# ============================================================================

print("\n\n================= 9. RESULTS AND EVALUATION =================")

# Create a function to evaluate models
def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    return {
        'Model': model_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

# Evaluate all models on the full dataset
models = {
    'Simple Linear Regression': (model_simple, X_simple, y),
    'Multiple Linear Regression': (model_multi, X_multi, y_multi),
    'Multiple Linear Regression with Seasonality': (model_multi_seasonal, X_multi_seasonal, y_multi),
    'Polynomial Regression': (model_poly, X_poly, y)
}

results = []
for name, (model, X, y_data) in models.items():
    results.append(evaluate_model(model, X, y_data, name))

results_df = pd.DataFrame(results)
print("Model Comparison Table:")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Residual Analysis
plt.figure(figsize=(14, 10))

# Plot for Simple Linear Regression
plt.subplot(2, 2, 1)
y_pred_simple = model_simple.predict(X_simple)
residuals_simple = y - y_pred_simple
plt.scatter(y_pred_simple, residuals_simple)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Simple Linear Regression Residuals')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

# Plot for Polynomial Regression
plt.subplot(2, 2, 2)
y_pred_poly = model_poly.predict(X_poly)
residuals_poly = y - y_pred_poly
plt.scatter(y_pred_poly, residuals_poly)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Polynomial Regression Residuals')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

# For Multiple Regression with lags
plt.subplot(2, 2, 3)
y_pred_multi = model_multi.predict(X_multi)
residuals_multi = y_multi - y_pred_multi
plt.scatter(y_pred_multi, residuals_multi)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Multiple Regression with Lags Residuals')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

# Residual Histogram
plt.subplot(2, 2, 4)
plt.hist(residuals_simple, bins=10, alpha=0.5, label='Simple')
plt.hist(residuals_poly, bins=10, alpha=0.5, label='Polynomial')
plt.title('Residual Histograms')
plt.legend()

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300)
plt.close()

print("\nResidual Analysis: Completed and visualized")

# Feature Importance (for simple model)
plt.figure(figsize=(8, 6))
importance = pd.DataFrame({
    'Feature': ['Crude Oil Price'],
    'Coefficient': model_simple.coef_
})
sns.barplot(x='Feature', y='Coefficient', data=importance)
plt.title('Feature Importance: Effect of Crude Oil Price on Gasoline Price')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()

print("\nFeature Importance: Analyzed and visualized")

# Prediction vs Actual Plot
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], y, 'o-', label='Actual Gas Price', color='blue')
plt.plot(df['Date'], model_simple.predict(X_simple), 's--', label='Predicted (Simple)', color='red')
plt.plot(df['Date'], model_poly.predict(X_poly), 'd-.', label='Predicted (Polynomial)', color='green')
plt.axvline(x=df['Date'].iloc[train_size], color='black', linestyle='--', 
            label='Train/Test Split')
plt.title('Actual vs Predicted Gasoline Prices')
plt.xlabel('Date')
plt.ylabel('Price (PHP/L)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('prediction_vs_actual.png', dpi=300)
plt.close()

print("\nPrediction vs Actual Plot: Created to visualize model performance")

# Create interactive Plotly visualization for added value
fig = make_subplots(rows=2, cols=1, 
                   shared_xaxes=True,
                   vertical_spacing=0.1,
                   subplot_titles=('Gasoline and Crude Oil Prices Over Time', 
                                  'Price Differential (Gasoline - Crude Oil)'))

# Add traces for prices
fig.add_trace(
    go.Scatter(x=df['Date'], y=df['Gas_Price'], name="Gasoline Price", 
              line=dict(color='blue', width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df['Date'], y=df['Oil_Price'], name="Crude Oil Price", 
              line=dict(color='red', width=2)),
    row=1, col=1
)

# Add trace for price differential
fig.add_trace(
    go.Scatter(x=df['Date'], y=df['Price_Diff'], name="Price Differential", 
              line=dict(color='green', width=2)),
    row=2, col=1
)

# Add predicted values
fig.add_trace(
    go.Scatter(x=df['Date'], y=model_simple.predict(X_simple), 
              name="Predicted (Simple Linear)", line=dict(color='orange', width=2, dash='dash')),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df['Date'], y=model_poly.predict(X_poly), 
              name="Predicted (Polynomial)", line=dict(color='purple', width=2, dash='dot')),
    row=1, col=1
)

# Update layout
fig.update_layout(
    title="Interactive Analysis of Gasoline and Crude Oil Prices (2023-2024)",
    height=800,
    legend_title="Price Series",
    hovermode="x unified"
)

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

fig.write_html("interactive_analysis.html")
print("\nCreated interactive Plotly visualization for presentation")

print("\n============================================================")
print("Analysis complete! All visualizations saved to disk.")
print("============================================================")