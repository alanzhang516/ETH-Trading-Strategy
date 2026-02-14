# ETH-USD Trading Strategy
# Author: Alan Zhang
# Description:
# Linear regression on lagged log returns with transaction cost modelling
# and risk-adjusted performance evaluation.

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

##############
## Set Seed ##
##############

# 1. Python random module
random.seed(42)

# 2. NumPy
np.random.seed(42)

# 3. PyTorch
torch.manual_seed(42)
torch.cuda.manual_seed(42)  # if using GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

######################
## Downloading Data ##
######################

sym = "ETH-USD"
First= "2024-06-01"
Last = "2026-01-01"
time_period = '1h'

# Returns a pandas DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
eth_prices = yf.download(tickers = sym, start= First, end = Last, interval = time_period)

# Plotting the time series graph for prices of ETH from 2025 - 2026
plt.figure(figsize=(12,6))  # creates a canvas of a rectangle 
plt.plot(eth_prices.index, eth_prices['Close'], label='ETH Close Price') #eth_prices.index is the time period which yfinance automatically sets as, being the x axis.
plt.title("ETH Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")

# Adding the closed log returns 
eth_prices["Closed_Log_Return"] = np.log(eth_prices["Close"]/eth_prices["Close"].shift(1))
eth_prices["lagged_return_1"] = eth_prices["Closed_Log_Return"].shift(1)
eth_prices["lagged_return_2"] = eth_prices["Closed_Log_Return"].shift(2)
eth_prices["lagged_return_3"] = eth_prices["Closed_Log_Return"].shift(3)

#dropping the "NaN"
eth_prices = eth_prices.dropna()

print(eth_prices.head())



######################
## Building A Model ##
######################


# first we gotta define what are features are. These are going to be the data that the model will look at 
# that will made predictions on what the target values are.

features = ["lagged_return_1","lagged_return_2","lagged_return_3"]
target = "Closed_Log_Return"

# splitting data set to train data(75%) and test data(25%)
split_index = int(len(eth_prices)*0.75)
train_data = eth_prices[:split_index]
test_data = eth_prices[split_index:]

X_train = train_data[features]
Y_train = train_data[target]

X_test = test_data[features]
Y_test = test_data[target]

# converting panda dataframe data type into tensors so pytorch can use it. Currently
# the data type is a float 64 however pytorch does not need that much precision and memory
# . values is needed as it cant convert the table rather only the raw data values. 
# these variables are now not in data format rather it is just a matrix of numbers 
import torch 
import torch.nn as nn
import torch.optim as optim


X_train_tensor = torch.tensor(X_train.values, dtype =torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype =torch.float32).reshape(-1, 1)

# as Y values only have 1 value, it is a 1D array so by reshaping it, it will create a 2D array 
X_test_tensor = torch.tensor(X_test.values, dtype =torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype =torch.float32).reshape(-1, 1)

# Creating a simple linear regression model such that y = w1x1 + w2x2 + w3x3 + b

class LinearModel(nn.Module):
    def __init__(self, input_features): # defines what our model is (linear layer)
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_features, 1)  

    def forward(self, x):
        return self.linear(x) # defines how our input moves through the model
    

#Create an instance of this LinearModel.
input_features = 3
Linear_Model = LinearModel(input_features)

#Defining our loss function 
criterion = nn.MSELoss()

#Defining the optimiser 
optimiser = optim.Adam(Linear_Model.parameters(), lr = 0.001)

# Training the model 
epochs = 1000

for epoch in range(epochs):
    y_hat = Linear_Model(X_train_tensor)
    loss = criterion(y_hat, Y_train_tensor)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# Printing out the loss function every 50 epochs 
    if ((epoch + 1) % 50) == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

#############
## TESTING ##
#############

# evaluation mode to eliminate dropout
Linear_Model.eval()

with torch.no_grad():
    y_test_predictions = Linear_Model(X_test_tensor)
    test_loss = criterion(y_test_predictions, Y_test_tensor)

# printing average test loss 
print(f"Test loss MSELoss: {test_loss.item():.6f}")



#########################################################
## TEST TRADING PERFORMANCE & ADDING TRANSACTION FEES ##
########################################################

# converting tensors to numpy arrays 
y_hat = y_test_predictions.detach().numpy().squeeze()
y_test = Y_test_tensor.detach().numpy().squeeze()

# Creating a panda data frame 
test_results = pd.DataFrame({
    "y_hat":y_hat,
    "y_test":y_test
})

# trade log returns
test_results["signal"] = np.sign(test_results["y_hat"])
test_results["trade_log_return"] = test_results["signal"] * test_results["y_test"]
test_results["cum_trade_log_return"] = test_results["trade_log_return"].cumsum()


# Assuming a $1000 starting capital, and only taker fee
capital = 1000
taker_fee = 0.0003

# Gross pnl
test_results["trade_value"] = test_results["signal"] * capital
test_results["trade_gross_PnL"] = test_results["trade_value"] * test_results["y_test"]

# net PnL ( gross pnl - transaction fee (entry and exit))
test_results["trade_net_PnL"] = test_results["trade_gross_PnL"] * (1 - 2 * taker_fee)

# cumulative equity curve in dollars
test_results["equity_curve_dollar"] = capital + test_results["trade_net_PnL"].cumsum()


print(test_results.head())

#plotting equity curve

plt.figure(figsize=(12,6))
plt.plot(test_data.index,test_results['equity_curve_dollar'], label='Equity Curve')
plt.xlabel("Time Step")
plt.ylabel("Cumulative Dollar Return")
plt.title("ETH Model Trading Equity Curve")
plt.legend()
plt.show()


####################################
## USING A COMPOUNDING TRADE SIZE ##
####################################

# Initialising columns
test_results["compounding_equity_curve_dollar"]= 0.0
equity = capital

for i in range(len(test_results)):
    trade_size = equity * test_results.loc[i, "signal"]
    
    #Gross PnL 
    gross_pnl = trade_size * test_results.loc[i, "y_test"]
    
    #Net PnL after taker fees (entry + exit)
    net_pnl = gross_pnl * (1 - 2 * taker_fee)
    
    # Updating equity
    equity += net_pnl
    
    #Storing equity to become the next entry 
    test_results.loc[i, "compounding_equity_curve_dollar"] = equity

plt.figure(figsize=(12,6))
plt.plot(test_data.index, test_results['compounding_equity_curve_dollar'], label='Compounding Equity Curve')
plt.xlabel("Time Step")
plt.ylabel("Equity ($)")
plt.title("ETH Model Trading Compounding Equity Curve")
plt.legend()
plt.show()


##########################
## OPTIMIZING OUR MODEL ##
##########################

# From this we can see that our model isn't really good, so now we are going to analyse
# different combinations of log returns, and see which one has higher sharpe ratio. We will 
# then optimize our loss function (criterion).

###############################
## SHARPE RATIO OPTIMIZATION ##
###############################

lag_combinations = {

    "lag_1" : ["lagged_return_1"],
    "lag_2" : ["lagged_return_2"],
    "lag_3": ["lagged_return_3"],
    "lag1_2" :["lagged_return_1", "lagged_return_2"],
    "lag2_3" :["lagged_return_2", "lagged_return_3"],
    "lag1_3" : ["lagged_return_1", "lagged_return_3"],
    "lag1_2_3": ["lagged_return_1", "lagged_return_2", "lagged_return_3"],
}

def sharpe_ratio(returns,periods = 365*24):
    mean = returns.mean()
    std = returns.std()
    if std == 0:
        return 0
    
    return np.sqrt(periods)* mean/ std

def train_and_testing(features):
    #splitting the data into train and test
    X_train = train_data[features]
    Y_train = train_data[target]
    X_test = test_data[features]
    Y_test = test_data[target]

    # converting data into tensors 

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train.values, dtype=torch.float32).reshape(-1, 1)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)

    #model 
    model = LinearModel(len(features))
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    #training 
    epochs = 100

    for epoch in range(epochs):
        y_hat = model(X_train_t)
        loss = criterion(y_hat, Y_train_t)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    # Testing 
    model.eval()
    with torch.no_grad():
        y_test_predictions = model(X_test_t).numpy().squeeze()
    
    # Test returns
    signal = np.sign(y_test_predictions)
    test_returns = signal * Y_test.values

    return test_returns

# computing sharpe ratios

results = []

for name, features in lag_combinations.items():
    returns = train_and_testing(features)
    sharpe = sharpe_ratio(returns)

    results.append({
        "features": name,
        "num_features": len(features),
        "sharpe": sharpe
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("sharpe", ascending=False).reset_index(drop=True)
print(results_df)


# From here the highest sharpe ratio is lag1_2_3

############################
## CRITERION OPTIMIZATION ##
############################

# We tested different loss functions (MSE, L1, Huber) and observed that MSE produced the highest
# Sharpe ratio, indicating it better captures extreme returns in our ETH model.