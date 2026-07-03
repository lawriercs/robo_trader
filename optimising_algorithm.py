from matplotlib import ticker
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker_pool = ["AAPL", "MSFT"]

print("Downloading data...")
raw_data = yf.download(ticker_pool + ["SPY"], start = "2024-08-01", end = "2026-06-02", interval="1h")

def optimise_parameters(ticker, buy_options, sell_options, adx_options, stop_loss_options, profit_target_options, start_idx, end_idx):
    def calculate_adx(high, low, close, period=14):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high.diff()
        down_move = -low.diff()


        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        pos_di = 100 * pos_dm.ewm(alpha=1/period, adjust=False).mean() / atr
        neg_di = 100 * neg_dm.ewm(alpha=1/period, adjust=False).mean() / atr
        
        dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx

    def backtest_strat(ticker, buy_thresh, sell_thresh, adx_thresh, stop_loss, profit_target, start_idx, end_idx):
        # Slice downloaded global data for this specific run
        close_prices = raw_data['Close'].copy().astype(float)
        
        if isinstance(close_prices.columns, pd.MultiIndex):
            close_prices.columns = [col[1] for col in close_prices.columns]

        high_prices_full = raw_data[('High', ticker)]
        low_prices_full = raw_data[('Low', ticker)]
        close_prices_full = raw_data[('Close', ticker)]

        adx_values_full = calculate_adx(high_prices_full, low_prices_full, close_prices_full, period=14)

        # Calculate EMAs
        ema_values_26_full = close_prices_full.ewm(span=26, adjust=False).mean()
        ema_values_12_full = close_prices_full.ewm(span=12, adjust=False).mean()
        ema_values_200_full = close_prices_full.ewm(span=200, adjust=False).mean()
        ema_values_50_full = close_prices_full.ewm(span=50, adjust=False).mean()

        close_prices_ticker = close_prices_full.iloc[start_idx:end_idx]
        adx_values = adx_values_full.iloc[start_idx:end_idx]
        ema_values_12 = ema_values_12_full.iloc[start_idx:end_idx]
        ema_values_26 = ema_values_26_full.iloc[start_idx:end_idx]

        cash = 10000.0
        position = 0               
        purchase_price = 0.0      
        max_price = 0.0
        portfolio_value = [] 
        
        for i in range(start_idx, end_idx):
            close_price = close_prices_full.iloc[i]
            adx_today = adx_values_full.iloc[i]

            if i < 200:
                portfolio_value.append(cash)
                continue

            e12_today = ema_values_12_full.iloc[i]
            e26_today = ema_values_26_full.iloc[i]
            e200_today = ema_values_200_full.iloc[i]
            e50_today = ema_values_50_full.iloc[i]

            ema_spread_12_26 = (e12_today - e26_today) / e26_today

            # --- FIX 2: STATE FILTER INSTEAD OF ON-THE-HOUR CROSSOVER ---
            ema_buy_signal = (ema_spread_12_26 > buy_thresh) 
            ema_sell_signal = (ema_spread_12_26 < sell_thresh) 

            if ema_buy_signal and adx_today >= adx_thresh and position == 0:
                position = int(cash // close_price)
                purchase_price = close_price
                max_price = close_price
                cash -= position * close_price

            elif position > 0:
                if close_price > max_price:
                    max_price = close_price  

                trailing_floor = max_price * stop_loss

                if close_price < trailing_floor:
                    cash += position * close_price
                    position = 0
                elif close_price >= purchase_price * profit_target:
                    cash += position * close_price
                    position = 0
                elif ema_sell_signal and adx_today >= adx_thresh:
                    cash += position * close_price
                    position = 0
                
            portfolio_value.append(cash + (position * close_price))

        portfolio_start_value = portfolio_value[200]
        portfolio_return = ((portfolio_value[-1] / portfolio_start_value) - 1) * 100
        return portfolio_return

    # Optimization inputs
    buy_options = [0.001,  0.005,  0.010]
    sell_options = [-0.01,  -0.05,  -0.10]
    adx_options = [ 15, 20, 25]
    stop_loss_options = [0.90, 0.93, 0.95, 0.97, ]
    profit_target_options = [ 1.05, 1.08, 1.10, ]
#"
    test_tickers = ticker_pool
    best_results_by_stock = {}

    for ticker in test_tickers:
        print(f"Optimizing parameters for {ticker}...")
        best_return = -100.0  # Set low so any gain beats it
        best_params = {}
        
        for b in buy_options:
            for s in sell_options:
                for a in adx_options:
                    for l in stop_loss_options:
                        for p in profit_target_options:
                            final_val = backtest_strat(ticker=ticker, buy_thresh=b, sell_thresh=s, adx_thresh=a, stop_loss=l, profit_target=p, start_idx=start_idx, end_idx=end_idx)

                            if final_val > best_return:
                                best_return = final_val
                                best_params = {'buy_spread': b, 'sell_spread': s, 'adx_cutoff': a, 'stop_loss': l, 'profit_target': p}
                        
        best_results_by_stock[ticker] = {
            "final_value": best_return,
            "parameters": best_params
        }
    return best_results_by_stock, best_params 


results = optimise_parameters(
    ticker=ticker_pool,
    buy_options=[0.001, 0.005, 0.01], 
    sell_options=[-0.01, -0.05, -0.1], 
    adx_options=[15, 20, 25], 
    stop_loss_options=[0.90, 0.93, 0.95, 0.97], 
    profit_target_options=[1.05, 1.08, 1.10], 
    start_idx=200, 
    end_idx=len(raw_data)
)

# --- FIX 3: Restructured clean terminal printing routine outside the function scope ---
print("\n--- OPTIMIZATION SUMMARY ---")
for stock, data in results.items():
    print(f"{stock}: Max Return {data['final_value']:.2f}% using {data['parameters']}")
    