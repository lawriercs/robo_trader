import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. DEFINE POOL & DOWNLOAD HIGH-RESOLUTION INTRADAY DATA
# -------------------------------------------------------------------
ticker_pool = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "COST",
    "INTC", "CSCO", "CMCSA", "PEP", "ADBE", "QCOM", "TXN", "AMGN", "HON", "AMAT",
    "SBUX", "BKNG", "VRTX", "MDLZ", "GILD", "REGN", "LRCX", "PANW", "SNPS", "KLAC",
    "CDNS", "ASML", "MELI", "MAR", "ORLY", "CTAS", "NXPI", "WDAY", "MNST", "LULU",
    "JPM", "BAC", "WMT", "DIS", "XOM", "CVX", "UNH", "HD", "V", "MA", "PG", "ABV",
    "LLY", "MRK", "PFE", "T", "VZ", "KO", "ORCL", "CRM", "NKE", "ADSK", "AAL", "DAL",
    "UAL", "BA", "CAT", "DE", "GE", "MMM", "F", "GM", "UBER", "ABNB", "COIN", "PLTR"]

print("Downloading high-resolution 1-minute intraday data...")
# 1-minute data has a max history of 60 days on yfinance. We will pull 10 days.
raw_data = yf.download(ticker_pool + ["SPY"], period="730d", interval="4h")

close_prices = raw_data['Close'].copy()
volumes = raw_data['Volume'].copy()

if isinstance(close_prices.columns, pd.MultiIndex):
    close_prices.columns = [col[1] for col in close_prices.columns]
    volumes.columns = [col[1] for col in volumes.columns]

# Drop rows where all elements are NaN (e.g., pre/post market hours if not populated)
close_prices.dropna(how='all', inplace=True)
volumes.dropna(how='all', inplace=True)

# -------------------------------------------------------------------
# 2. INTRADAY INDICATOR ENGINE
# -------------------------------------------------------------------
print("Calculating short-term intraday volume moving averages...")
# Shorter window (10 periods) to quickly adapt to changing intraday volume cycles
avg_volume_10m = volumes.rolling(window=10).mean()

trade_pool = [t for t in ticker_pool if t in close_prices.columns]
timestamps = close_prices.index

# -------------------------------------------------------------------
# 3. DAY TRADING SIMULATION RUNNER
# -------------------------------------------------------------------
initial_cash = 10000.00
cash = initial_cash
max_positions = 5  # Concentrating capital into fewer simultaneous day trades
active_positions = {} 
portfolio_history = []

for i in range(1, len(timestamps)):
    current_time = timestamps[i]
    prev_time = timestamps[i-1]
    
    # Extract current hour and minute for EOD rules
    current_hour = current_time.hour
    current_minute = current_time.minute
    
    # Calculate Total Portfolio Value for this specific 5-minute bar
    current_portfolio_value = cash
    for tkr, pos in active_positions.items():
        current_portfolio_value += pos["shares"] * float(close_prices.at[current_time, tkr])
    portfolio_history.append(current_portfolio_value)
    
    allocation_per_trade = current_portfolio_value / max_positions

    # A. MONITOR & MANAGE OPEN DAY TRADES
    tickers_to_remove = []
    for tkr, pos in active_positions.items():
        price = float(close_prices.at[current_time, tkr])
        prev_price = float(close_prices.at[prev_time, tkr])
        vol = float(volumes.at[current_time, tkr])
        avg_vol = float(avg_volume_10m.at[current_time, tkr])
        
        if np.isnan(price) or np.isnan(vol) or np.isnan(avg_vol):
            continue

        # RULE 1: Emergency Hard Time Cutoff (Liquidate at 3:50 PM EST / 15:50)
        # Market closes at 16:00 EST. We exit 10 mins before to avoid closing bells spreads.
        is_market_closing = (current_hour >= 11 ) or (current_hour >= 16)
        
        # RULE 2: Standard Volume Spike Exit
        volume_exit = (price < prev_price) and (vol > avg_vol * 2.5)

        if is_market_closing:
            cash += pos["shares"] * price
            tickers_to_remove.append(tkr)
            print(f"{current_time.strftime('%Y-%m-%d %H:%M')}: [CLOCK EXIT] Forced EOD Close for {tkr} at ${price:.2f}")
        elif volume_exit:
            cash += pos["shares"] * price
            tickers_to_remove.append(tkr)
            print(f"{current_time.strftime('%Y-%m-%d %H:%M')}: [VOLUME EXIT] Momentum shift on {tkr} at ${price:.2f}")

    for tkr in tickers_to_remove:
        del active_positions[tkr]

    # B. DAY TRADE SYSTEMATIC ENTRY
    # Do not enter new positions if we are within 45 minutes of market close
    allowed_to_enter = not ((current_hour == 15 and current_minute >= 15) or (current_hour >= 16))

    if len(active_positions) < max_positions and allowed_to_enter:
        for tkr in trade_pool:
            if tkr in active_positions or len(active_positions) >= max_positions:
                continue
                
            price = float(close_prices.at[current_time, tkr])
            prev_price = float(close_prices.at[prev_time, tkr])
            vol = float(volumes.at[current_time, tkr])
            avg_vol = float(avg_volume_10m.at[current_time, tkr])
            
            if np.isnan(price) or np.isnan(vol) or np.isnan(avg_vol):
                continue
                
            # Quick aggressive spike trigger for intraday scalps
            volume_buy = (price > prev_price) and (vol > avg_vol * 2.5)
            
            if volume_buy and cash >= allocation_per_trade:
                shares = int(allocation_per_trade // price)
                if shares > 0:
                    cash -= shares * price
                    active_positions[tkr] = {
                        "shares": shares, 
                        "entry_price": price
                    }
                    print(f"{current_time.strftime('%Y-%m-%d %H:%M')}: [DAY TRADE ENTER] Scalping {tkr} at ${price:.2f}")

# Final portfolio evaluation point
portfolio_history.append(cash + sum(pos["shares"] * float(close_prices.iloc[-1][tkr]) for tkr, pos in active_positions.items()))

# -------------------------------------------------------------------
# 4. PLOT INTRADAY RESULTS
# -------------------------------------------------------------------
final_val = portfolio_history[-1]
spy_return = ((close_prices['SPY'].iloc[-1] - close_prices['SPY'].iloc[0]) / close_prices['SPY'].iloc[0]) * 100
strat_return = ((final_val - initial_cash) / initial_cash) * 100

print("-" * 75)
print(f"Day Trading Strategy Return: {strat_return:.2f}% | SPY Return over same period: {spy_return:.2f}%")

plt.figure(figsize=(12, 6))
spy_normalized = (close_prices['SPY'] / close_prices['SPY'].iloc[0]) * initial_cash

plt.plot(close_prices.index[:len(portfolio_history)], portfolio_history, label='Intraday Volume Day Trader', color='dodgerblue', linewidth=2)
plt.plot(spy_normalized.index, spy_normalized, label='S&P 500 (SPY)', color='gray', linestyle='--', alpha=0.7)
plt.title('Quick Intraday Volume Spike Day Trading vs SPY')
plt.xlabel('Date/Time')
plt.ylabel('Total Capital ($)')
plt.grid(True, linestyle=':')
plt.legend()
plt.show()