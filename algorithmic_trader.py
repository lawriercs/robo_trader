import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define pool of tickers
ticker_pool = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "COST",
    #"INTC", "CSCO", "CMCSA", "PEP", "ADBE", "QCOM", "TXN", "AMGN", "HON", "AMAT",
    #"SBUX", "BKNG", "VRTX", "MDLZ", "GILD", "REGN", "LRCX", "PANW", "SNPS", "KLAC",
    #"CDNS", "ASML", "MELI", "MAR", "ORLY", "CTAS", "NXPI", "WDAY", "MNST", "LULU",
    #"JPM", "BAC", "WMT", "DIS", "XOM", "CVX", "UNH", "HD", "V", "MA", "PG", "ABBV",
    #"LLY", "MRK", "PFE", "T", "VZ", "KO", "ORCL", "CRM", "NKE", "ADSK", "AAL", "DAL",
    #"UAL", "BA", "CAT", "DE", "GE", "MMM", "F", "GM", "UBER", "ABNB", "COIN", "PLTR"]
]

# Download data for the defined pool of tickers and SPY as a benchmark
raw_data = yf.download(ticker_pool + ["SPY"], period="300d", interval="1h")

# Create a dataset of close prices and volumes
close_prices = raw_data['Close'].copy().astype(float)

# Allows for the calling of columns without the MultiIndex tuple if present
if isinstance(close_prices.columns, pd.MultiIndex):
    close_prices.columns = [col[1] for col in close_prices.columns]

def backtest_strategy(ticker):

    # Calculate a short and a longer EMA
    ema_values_12 = close_prices[ticker].ewm(span=12, adjust=False).mean()
    ema_values_26 = close_prices[ticker].ewm(span=26, adjust=False).mean()
    ema_values_50 = close_prices[ticker].ewm(span=50, adjust=False).mean()
    ema_values_200 = close_prices[ticker].ewm(span=200, adjust=False).mean()

    # Simulating a simple trading strategy based on EMA crossovers
    cash = 10000.0
    position = 0               
    purchase_price = 0.0      
    portfolio_value = [] 
    trailing_floor_history = []  

    # Loop using an index 
    for i in range(len(close_prices)):
        close_price = close_prices[ticker].iloc[i]
        
        # Since i = 0 the first iteration is skipped
        if i < 200:
            portfolio_value.append(cash)
            continue

        # Calculate the signals inside the loop for day 'i'
        e12_today = ema_values_12.iloc[i]
        e26_today = ema_values_26.iloc[i]
        #e12_yest = ema_values_12.iloc[i-1]
        #e26_yest = ema_values_26.iloc[i-1]
        e200_today = ema_values_200.iloc[i]
        #e200_yest = ema_values_200.iloc[i-1]
        e50_today = ema_values_50.iloc[i]
        #e50_yest = ema_values_50.iloc[i-1]

        ema_spread_12_26 = (e12_today - e26_today) / e26_today

        #ema_buy_signal = (e12_today > e200_today) and (e12_yest < e200_yest)
        ema_buy_signal = (ema_spread_12_26 > 0.01) and (e12_today >e26_today > e50_today > e200_today) and (e200_today > 0) and (e50_today > 0)
        ema_sell_signal = (ema_spread_12_26 < -0.005) and (e12_today < e26_today < e50_today < e200_today)

        # Execution Logic
        if ema_buy_signal and position == 0:
            position = int(cash // close_price)
            purchase_price = close_price
            max_price = close_price
            cash -= position * close_price
            print(f"Bought {position} shares at {close_price:.2f}") 

        elif position > 0:
            if close_price > max_price:
                max_price = close_price  

            trailing_floor = max_price * 0.93

            if close_price < trailing_floor:
                cash += position * close_price
                print(f"Trailing stop triggered. Sold at {close_price:.2f} | Peak was {max_price:.2f}")
                position = 0
                purchase_price = 0.0
                max_price = 0.0

            elif ema_sell_signal and position > 0:
                cash += position * close_price
                print(f"Sold {position} shares at {close_price:.2f} | Profit: {(close_price - purchase_price) * position:.2f}")
                position = 0
                purchase_price = 0.0
            
        # Track portfolio value at the end of every single step
        portfolio_value.append(cash + (position * close_price))
        
        if position > 0:
            trailing_floor_history.append(max_price * 0.93)
        else:
            trailing_floor_history.append(np.nan)

    # Find the starting values at index 200 where the trading loop begins
    stock_start = close_prices[ticker].iloc[200]
    strat_start = portfolio_value[200]
    stock_end = close_prices[ticker].iloc[-1]
    stock_perf = ((stock_end / stock_start) - 1) * 100
    strat_end = portfolio_value[-1]
    strat_perf = ((strat_end / strat_start) - 1) * 100

    return strat_perf, stock_perf, portfolio_value, trailing_floor_history, strat_start

results = []
all_portfolios = []  # To accumulate each asset's performance array

for t in ticker_pool:
    print(f"Crunching numbers for {t}...")
    strat_return, stock_return, p_val, _, s_start = backtest_strategy(t)
    
    results.append({
        "Ticker": t,
        "Strategy Return (%)": round(strat_return, 2),
        "Stock Return (%)": round(stock_return, 2),
        "Outperformed Stock?": "Yes" if strat_return > stock_return else "No"
    })
    
    # Normalize this single asset's tracking curve and save it
    norm_p = (np.array(p_val) / s_start) * 100
    all_portfolios.append(norm_p)

# Benchmark calculation
spy_start = close_prices["SPY"].iloc[200]
spy_perf = round(((close_prices["SPY"].iloc[-1] / spy_start) - 1) * 100, 2)
normalized_sp500 = (close_prices["SPY"] / spy_start) * 100

# Average all individual normalized curves to get your TRUE composite portfolio performance
mean_portfolio_curve = np.mean(all_portfolios, axis=0)
    
df_results = pd.DataFrame(results)
print("\n" + "="*55)
print(f" PERFORMANCE LEADERBOARD (Benchmark SPY: {spy_perf}%)")
print("="*55)
print(df_results.to_string(index=False))
print("="*55)
print(f"Average Strategy Return: {round(df_results['Strategy Return (%)'].mean(), 2)}% | S&P 500 Return: {round(spy_perf, 2)}%")

# Create the corrected chart
plt.figure(figsize=(12, 6))
# Slice [200:] so it properly aligns to your post-warmup starting point
plt.plot(close_prices.index[200:], mean_portfolio_curve[200:], label='Total Strategy Portfolio (%)', color='orange', linewidth=2)
plt.plot(close_prices.index[200:], normalized_sp500[200:], label='S&P 500 Benchmark (%)', color='blue', linewidth=2)
plt.xlabel("Date")
plt.ylabel("Normalized Growth (Starts at 100%)")
plt.title("True Aggregated Strategy Performance vs S&P 500")
plt.legend()
plt.grid()
plt.show()
