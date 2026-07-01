import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. DEFINE POOL & DOWNLOAD BROAD MARKET DATA (SPY)
# -------------------------------------------------------------------
ticker_pool = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "COST",
    "INTC", "CSCO", "CMCSA", "PEP", "ADBE", "QCOM", "TXN", "AMGN", "HON", "AMAT",
    "SBUX", "BKNG", "VRTX", "MDLZ", "GILD", "REGN", "LRCX", "PANW", "SNPS", "KLAC",
    "CDNS", "ASML", "MELI", "MAR", "ORLY", "CTAS", "NXPI", "WDAY", "MNST", "LULU",
    "JPM", "BAC", "WMT", "DIS", "XOM", "CVX", "UNH", "HD", "V", "MA", "PG", "ABV",
    "LLY", "MRK", "PFE", "T", "VZ", "KO", "ORCL", "CRM", "NKE", "ADSK", "AAL", "DAL",
    "UAL", "BA", "CAT", "DE", "GE", "MMM", "F", "GM", "UBER", "ABNB", "COIN", "PLTR"]
all_tickers = ticker_pool + ["SPY"]

print("Downloading data and calculating Alpha metrics...")
raw_data = yf.download(all_tickers, period="2y", interval="1h")
raw_data.columns = [col[0] if isinstance(col, tuple) else col for col in raw_data.columns] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data.columns
# Clean dataframes
close_prices = raw_data['Close']
volumes = raw_data['Volume']

# -------------------------------------------------------------------
# 1. DEFINE POOL & DOWNLOAD DATA INDEPENDENTLY FOR SAFETY
# -------------------------------------------------------------------
ticker_pool = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "COST"]

print("Downloading trading pool and market benchmark...")
# We download everything explicitly
raw_data = yf.download(ticker_pool + ["SPY"], period="2y", interval="1h")

# Extract close and volume matrices directly from the raw download structure
close_prices = raw_data['Close'].copy()
volumes = raw_data['Volume'].copy()

# -------------------------------------------------------------------
# 2. INDICATOR ENGINE (REGIME + MOMENTUM + VOLATILITY)
# -------------------------------------------------------------------
# Calculate SPY filter directly from the clean close dataframe
spy_sma = close_prices['SPY'].rolling(window=200).mean()

print("Calculating technical indicators across the portfolio...")
# Calculate indicators across all columns simultaneously
avg_volume_20h = volumes.rolling(window=20).mean()

price_changes = close_prices.diff()
gains = price_changes.clip(lower=0)
losses = -price_changes.clip(upper=0)

avg_gains = gains.ewm(com=13, adjust=False).mean()
avg_losses = losses.ewm(com=13, adjust=False).mean()
rs = avg_gains / avg_losses
rsi_all = 100 - (100 / (1 + rs))

# Ensure our execution code knows exactly what columns belong to real stocks
trade_pool = [t for t in ticker_pool if t in close_prices.columns]
timestamps = close_prices.index

# -------------------------------------------------------------------
# 3. HIGH-ALPHA SIMULATION RUNNER
# -------------------------------------------------------------------
initial_cash = 10000.00
cash = initial_cash
max_positions = 10
active_positions = {} # Structure: {tkr: {"shares": X, "highest_price": Y}}
portfolio_history = []

for i in range(1, len(timestamps)):
    current_time = timestamps[i]
    prev_time = timestamps[i-1]
    
    # Global Switch Check
    market_is_bullish = float(close_prices.at[current_time, 'SPY']) > float(spy_sma.at[current_time])
    
    # Calculate Total Portfolio Net Asset Value for this hour
    current_portfolio_value = cash
    for tkr, pos in active_positions.items():
        current_portfolio_value += pos["shares"] * float(close_prices.at[current_time, tkr])
    portfolio_history.append(current_portfolio_value)
    
    allocation_per_trade = current_portfolio_value / max_positions

    # A. MONITOR & MANAGE OPEN TRADES (Trailing Stops)
    tickers_to_remove = []
    for tkr, pos in active_positions.items():
        price = float(close_prices.at[current_time, tkr])
        prev_price = float(close_prices.at[prev_time, tkr])
        vol = float(volumes.at[current_time, tkr])
        avg_vol = float(avg_volume_20h.at[current_time, tkr])
        
        # Update trailing baseline if stock hit a new high since we bought it
        if price > pos["highest_price"]:
            pos["highest_price"] = price
            
        # Trailing Stop Floor set at 7% below the maximum peak reached
        trailing_stop_floor = pos["highest_price"] * 0.84
        # Initial Hard Stop Loss set at 5% below entry price
        hard_stop_floor = pos["entry_price"] * 0.05 # wait, let's look at standard 5% floor
        hard_stop_floor = pos["entry_price"] * 0.95

        # Check Exits
        if price <= trailing_stop_floor:
            cash += pos["shares"] * price
            tickers_to_remove.append(tkr)
            print(f"{current_time.strftime('%Y-%m-%d %H:%M')}: [TRAILING EXIT] {tkr} locked in at ${price:.2f}")
        elif price <= hard_stop_floor:
            cash += pos["shares"] * price
            tickers_to_remove.append(tkr)
            print(f"{current_time.strftime('%Y-%m-%d %H:%M')}: [HARD LOSS EXIT] {tkr} cut at ${price:.2f}")
        elif (price < prev_price) and (vol > avg_vol * 3):
            cash += pos["shares"] * price
            tickers_to_remove.append(tkr)
            print(f"{current_time.strftime('%Y-%m-%d %H:%M')}: [VOLUME EXIT] Institutional distribution on {tkr} at ${price:.2f}")

    for tkr in tickers_to_remove:
        del active_positions[tkr]

    # B. SYSTEMATIC ENTRY (Only if Broad Market is safe)
    if len(active_positions) < max_positions and market_is_bullish:
        for tkr in trade_pool:
            if tkr in active_positions or len(active_positions) >= max_positions:
                continue
                
            price = float(close_prices.at[current_time, tkr])
            prev_price = float(close_prices.at[prev_time, tkr])
            rsi = float(rsi_all.at[current_time, tkr])
            vol = float(volumes.at[current_time, tkr])
            avg_vol = float(avg_volume_20h.at[current_time, tkr])
            
            if np.isnan(price) or np.isnan(rsi) or np.isnan(vol):
                continue
                
            rsi_buy = rsi <= 35
            volume_buy = (price > prev_price) and (vol > avg_vol * 3)
            
            if (rsi_buy or volume_buy) and cash >= allocation_per_trade:
                shares = int(allocation_per_trade // price)
                if shares > 0:
                    cash -= shares * price
                    active_positions[tkr] = {
                        "shares": shares, 
                        "entry_price": price,
                        "highest_price": price
                    }
                    print(f"{current_time.strftime('%Y-%m-%d %H:%M')}: [ENTER] Loaded {tkr} at ${price:.2f}")

portfolio_history.append(cash + sum(pos["shares"] * float(close_prices.iloc[-1][tkr]) for tkr, pos in active_positions.items()))

# -------------------------------------------------------------------
# 4. PLOT ALPHA RESULTS VS BENCHMARK
# -------------------------------------------------------------------
final_val = portfolio_history[-1]
spy_return = ((close_prices['SPY'].iloc[-1] - close_prices['SPY'].iloc[0]) / close_prices['SPY'].iloc[0]) * 100
strat_return = ((final_val - initial_cash) / initial_cash) * 100

print("-" * 75)
print(f"Strategy Return: {strat_return:.2f}% | S&P 500 Market Return: {spy_return:.2f}%")

plt.figure(figsize=(12, 6))
# Normalize SPY to start at $10k to compare dollar-for-dollar performance
spy_normalized = (close_prices['SPY'] / close_prices['SPY'].iloc[0]) * initial_cash

plt.plot(close_prices.index[:len(portfolio_history)], portfolio_history, label='Upgraded Alpha Engine', color='forestgreen', linewidth=2)
plt.plot(spy_normalized.index, spy_normalized, label='S&P 500 Benchmark (SPY)', color='gray', linestyle='--', alpha=0.7)
plt.title('Alpha Engine vs S&P 500 Market Benchmark')
plt.xlabel('Date')
plt.ylabel('Total Asset Capital ($)')
plt.grid(True, linestyle=':')
plt.legend()
plt.show()