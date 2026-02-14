# ETH-USD  Trading Strategy

# Overview
This project implements a quantitative trading strategy on hourly ETH-USD data. The model predicts log returns using lagged returns as features and executes a
directional trading strategy based on predicted return sign.

# Model
- Linear regression implemented in PyTorch
- Optimized using Adam optimizer
- Loss function: MSE

# Backtesting
- Directional trading via sign(predicted return)
- Transaction cost: 0.03% taker fee
- Fixed capital and compounding frameworks tested

# Performance Metrics
- Sharpe Ratio
- Equity curve analysis

# Limitations:
This strategy was developed as a learning project and evaluated using historical backtesting. Past performance does not guarantee future results, and market conditions change over time. The model was also tested on a limited dataset, and performance may reflect some degree of overfitting. Overall, this project demonstrates the modelling and backtesting framework rather than a profitable and ready trading system.
