# Crude Oil and Gasoline Price Analysis

## Project Overview
This project analyzes the relationship between crude oil prices and retail gasoline prices in the Philippines from 2023 to 2024. The analysis includes statistical modeling, time series analysis, and interactive visualizations to understand how changes in crude oil prices affect gasoline prices at the pump.

## Key Features
- **Data Analysis**: Exploratory data analysis of 22 months of price data
- **Statistical Modeling**: Multiple regression models to quantify the relationship
- **Price Elasticity**: Calculation of price elasticity between crude oil and gasoline prices
- **Interactive Visualizations**: Dynamic charts to explore price trends
- **Model Comparison**: Evaluation of different regression techniques

## Key Findings
- The correlation between crude oil and gasoline prices is **0.3184**
- A ₱1 increase in crude oil price is associated with a **0.7611** PHP increase in gasoline price
- The best performing model is **Random Forest** with an R² of approximately **0.70**
- Average price elasticity is **0.6818**, indicating that gasoline prices change proportionally **less** than crude oil prices

## Data
The dataset includes monthly prices for:
- Gasoline prices (PHP/liter)
- Crude oil prices (PHP/liter)
- Time period: January 2023 - November 2024

## Analysis Methods
1. **Exploratory Data Analysis**
   - Time series visualization
   - Correlation analysis
   - Scatter plot analysis

2. **Statistical Modeling**
   - Simple linear regression
   - Multiple regression with lagged variables
   - Ridge and Lasso regression
   - Random Forest regression

3. **Advanced Analysis**
   - Price elasticity calculation
   - Time series decomposition
   - Hyperparameter tuning

## Visualizations
The project includes several visualizations:
- Price trend analysis
- Correlation heatmaps
- Regression results
- Residual analysis
- Model performance comparison
- Interactive plots

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- plotly

## How to Run
1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the analysis script: `python oil_price_analysis.py`

## Files and Directory Structure
- `oil_price_analysis.py`: Main analysis script
- `README.md`: Project documentation
- `price_trends.png`: Time series visualization of prices
- `price_relationship.png`: Scatter plot of price relationship
- `correlation_heatmap.png`: Correlation matrix visualization
- `regression_results.png`: Visualization of regression model results
- `residual_plot.png` & `residual_histogram.png`: Residual analysis
- `model_comparison_r2.png` & `model_comparison_rmse.png`: Model performance comparison
- `elasticity_analysis.png`: Price elasticity visualization
- `monthly_average_prices.png`: Monthly average prices
- `interactive_trends.html` & `interactive_scatter.html`: Interactive visualizations

## License
This project is licensed under the terms of the included LICENSE file.


## Acknowledgments
- Data collected from monthly price reports
- Special thanks to the research team for data collection and validation