import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define pool of tickers
ticker_pool = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "COST",
    "INTC", "CSCO", "CMCSA", "PEP", "ADBE", "QCOM", "TXN", "AMGN", "HON", "AMAT",
    "SBUX", "BKNG", "VRTX", "MDLZ", "GILD", "REGN", "LRCX", "PANW", "SNPS", "KLAC",
    "CDNS", "ASML", "MELI", "MAR", "ORLY", "CTAS", "NXPI", "WDAY", "MNST", "LULU",
    "JPM", "BAC", "WMT", "DIS", "XOM", "CVX", "UNH", "HD", "V", "MA", "PG", "ABBV",
    "LLY", "MRK", "PFE", "T", "VZ", "KO", "ORCL", "CRM", "NKE", "ADSK", "AAL", "DAL",
    "UAL", "BA", "CAT", "DE", "GE", "MMM", "F", "GM", "UBER", "ABNB", "COIN", "PLTR"]


ticker = "AAPL"  
# Download data for the defined pool of tickers and SPY as a benchmark
raw_data = yf.download(ticker_pool + ["SPY"], period="200d", interval="1h")

# Create a dataset of close prices and volumes
close_prices = raw_data['Close'].copy().astype(float)
volumes = raw_data['Volume'].copy().astype(float)

# Allows for the calling of columns without the MultiIndex tuple if present
if isinstance(close_prices.columns, pd.MultiIndex):
    close_prices.columns = [col[1] for col in close_prices.columns]
    volumes.columns = [col[1] for col in volumes.columns]

# Calculates change in price for each ticker and the corresponding gains and losses
price_changes = close_prices.diff()
gains = price_changes.clip(lower=0)
losses = -price_changes.clip(upper=0)

# Calculate a short and a longer EMA
ema_period_20 = 20
ema_values_20 = close_prices[ticker].ewm(span=ema_period_20, adjust=False).mean()
ema_period_9 = 9
ema_values_9 = close_prices[ticker].ewm(span=ema_period_9, adjust=False).mean()

# Calculating EMA crossover signals
ema_values_9_changes = ema_values_9.diff()
ema_gains_9 = ema_values_9_changes.clip(lower=0)
ema_losses_9 = -ema_values_9_changes.clip(upper=0)

ema_values_20_changes = ema_values_20.diff()
ema_gains_20 = ema_values_20_changes.clip(lower=0)
ema_losses_20 = -ema_values_20_changes.clip(upper=0)

# Simulating a simple trading strategy based on EMA crossovers
cash = close_prices[ticker].iloc[0] 
position = 0               
purchase_price = 0.0      
portfolio_value = [] 

# Loop using an index 
for i in range(len(close_prices)):
    close_price = close_prices[ticker].iloc[i]
    
    # Since i = 0 the first iteration is skipped
    if i == 0:
        portfolio_value.append(cash)
        continue

    # Calculate the signals inside the loop for day 'i'
    ema_buy_signal = (ema_values_9.iloc[i] > ema_values_20.iloc[i]) and (ema_values_9.iloc[i-1] < ema_values_20.iloc[i-1])
    ema_sell_signal = (ema_values_9.iloc[i] < ema_values_20.iloc[i]) and (ema_values_9.iloc[i-1] > ema_values_20.iloc[i-1])

    # Execution Logic
    if ema_buy_signal and position == 0:
        position = int(cash // close_price)
        purchase_price = close_price
        cash -= position * close_price
        print(f"Bought {position} shares at {close_price:.2f}") # Optional: see trades

    elif ema_sell_signal and position > 0:
        cash += position * close_price
        print(f"Sold {position} shares at {close_price:.2f} | Profit: {(close_price - purchase_price) * position:.2f}")
        position = 0
        purchase_price = 0.0
        
    # Track portfolio value at the end of every single step
    portfolio_value.append(cash + (position * close_price))

plt.figure(figsize=(12, 6))
plt.plot(close_prices.index, close_prices[ticker], label=ticker)
plt.plot(close_prices.index, portfolio_value, label='Portfolio Value', color='orange')
plt.xlabel('Date/Time')
plt.ylabel('Price')
plt.title(f'{ticker} Price Movement')
plt.legend()
plt.grid()
plt.show()