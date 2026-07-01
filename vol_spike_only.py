import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. DEFINE POOL & DOWNLOAD DATA 
# -------------------------------------------------------------------
ticker_pool = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "COST"]

print("Downloading trading pool and market benchmark...")
raw_data = yf.download(ticker_pool + ["SPY"], period="2y", interval="1h")

# Clean dataframes
close_prices = raw_data['Close'].copy()
volumes = raw_data['Volume'].copy()

# Fix MultiIndex columns if present
if isinstance(close_prices.columns, pd.MultiIndex):
    close_prices.columns = [col[1] for col in close_prices.columns]
    volumes.columns = [col[1] for col in volumes.columns]

# -------------------------------------------------------------------
# 2. INDICATOR ENGINE (VOLUME SPIKES ONLY)
# -------------------------------------------------------------------
print("Calculating volume moving averages...")
# Baseline 20-hour volume average
avg_volume_20h = volumes.rolling(window=20).mean()

trade_pool = [t for t in ticker_pool if t in close_prices.columns]
timestamps = close_prices.index

# -------------------------------------------------------------------
# 3. VOLUME-SPIKE SIMULATION RUNNER
# -------------------------------------------------------------------
initial_cash = 10000.00
cash = initial_cash
max_positions = 10
active_positions = {} # Structure: {tkr: {"shares": X, "entry_price": Y}}
portfolio_history = []

for i in range(1, len(timestamps)):
    current_time = timestamps[i]
    prev_time = timestamps[i-1]
    
    # Calculate Total Portfolio Net Asset Value for this hour
    current_portfolio_value = cash
    for tkr, pos in active_positions.items():
        current_portfolio_value += pos["shares"] * float(close_prices.at[current_time, tkr])
    portfolio_history.append(current_portfolio_value)
    
    allocation_per_trade = current_portfolio_value / max_positions

    # A. MONITOR OPEN TRADES FOR EXITS (Volume Spike Down)
    tickers_to_remove = []
    for tkr, pos in active_positions.items():
        price = float(close_prices.at[current_time, tkr])
        prev_price = float(close_prices.at[prev_time, tkr])
        vol = float(volumes.at[current_time, tkr])
        avg_vol = float(avg_volume_20h.at[current_time, tkr])
        
        if np.isnan(price) or np.isnan(vol) or np.isnan(avg_vol):
            continue

        # EXCLUSIVE EXIT TRIGGER: Volume spike (> 3x average) on a down hour
        volume_exit = (price < prev_price) and (vol > avg_vol * 3)

        if volume_exit:
            cash += pos["shares"] * price
            tickers_to_remove.append(tkr)
            print(f"{current_time.strftime('%Y-%m-%d %H:%M')}: [VOLUME EXIT] Institutional distribution on {tkr} at ${price:.2f}")

    for tkr in tickers_to_remove:
        del active_positions[tkr]

    # B. SYSTEMATIC ENTRY (Volume Spike Up)
    if len(active_positions) < max_positions:
        for tkr in trade_pool:
            if tkr in active_positions or len(active_positions) >= max_positions:
                continue
                
            price = float(close_prices.at[current_time, tkr])
            prev_price = float(close_prices.at[prev_time, tkr])
            vol = float(volumes.at[current_time, tkr])
            avg_vol = float(avg_volume_20h.at[current_time, tkr])
            
            if np.isnan(price) or np.isnan(vol) or np.isnan(avg_vol):
                continue
                
            # EXCLUSIVE ENTRY TRIGGER: Volume spike (> 3x average) on an up hour
            volume_buy = (price > prev_price) and (vol > avg_vol * 3)
            
            if volume_buy and cash >= allocation_per_trade:
                shares = int(allocation_per_trade // price)
                if shares > 0:
                    cash -= shares * price
                    active_positions[tkr] = {
                        "shares": shares, 
                        "entry_price": price
                    }
                    print(f"{current_time.strftime('%Y-%m-%d %H:%M')}: [ENTER] Loaded {tkr} at ${price:.2f}")

# Final portfolio evaluation point
portfolio_history.append(cash + sum(pos["shares"] * float(close_prices.iloc[-1][tkr]) for tkr, pos in active_positions.items()))

# -------------------------------------------------------------------
# 4. PLOT RESULTS VS BENCHMARK
# -------------------------------------------------------------------
final_val = portfolio_history[-1]
spy_return = ((close_prices['SPY'].iloc[-1] - close_prices['SPY'].iloc[0]) / close_prices['SPY'].iloc[0]) * 100
strat_return = ((final_val - initial_cash) / initial_cash) * 100

print("-" * 75)
print(f"Strategy Return: {strat_return:.2f}% | S&P 500 Market Return: {spy_return:.2f}%")

plt.figure(figsize=(12, 6))
spy_normalized = (close_prices['SPY'] / close_prices['SPY'].iloc[0]) * initial_cash

plt.plot(close_prices.index[:len(portfolio_history)], portfolio_history, label='Pure Volume Spike Strategy', color='forestgreen', linewidth=2)
plt.plot(spy_normalized.index, spy_normalized, label='S&P 500 Benchmark (SPY)', color='gray', linestyle='--', alpha=0.7)
plt.title('Pure Volume Spike Strategy vs S&P 500')
plt.xlabel('Date')
plt.ylabel('Total Asset Capital ($)')
plt.grid(True, linestyle=':')
plt.legend()
plt.show()