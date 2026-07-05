import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from optimising_algorithm import optimise_parameters

# Define pool of tickers
ticker_pool = ["AMD", 'GOOGL']  #, "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "COST",
    #"INTC", "CSCO", "CMCSA", "PEP", "ADBE", "QCOM", "TXN", "AMGN", "HON", "AMAT",
    #"SBUX", "BKNG", "VRTX", "MDLZ", "GILD", "REGN", "LRCX", "PANW", "SNPS", "KLAC",
    #"CDNS", "ASML", "MELI", "MAR", "ORLY", "CTAS", "NXPI", "WDAY", "MNST", "LULU",
    #"JPM", "BAC", "WMT", "DIS", "XOM", "CVX", "UNH", "HD", "V", "MA", "PG", "ABBV",
    #"LLY", "MRK", "PFE", "UAL", "BA", "CAT", "DE", "GE", "MMM", "F", "GM", "UBER", "ABNB", "COIN", "PLTR"]

# Download data for the defined pool of tickers and SPY as a benchmark
raw_data = yf.download(ticker_pool + ["SPY"], start = "2024-08-01", end = "2026-06-02", interval="1h")

for ticker in ticker_pool:

    # Create a dataset of close prices and volumes
    close_prices = raw_data['Close'].copy().astype(float)
    volumes = raw_data['Volume'].copy().astype(float)

    best_results_by_stock, best_params = optimise_parameters(
        ticker=[ticker],
        buy_options=[0.001, 0.005, 0.01], 
        sell_options=[-0.01, -0.05, -0.1], 
        adx_options=[15, 20, 25], 
        stop_loss_options=[0.90, 0.93, 0.95, 0.97], 
        profit_target_options=[1.05, 1.08, 1.10], 
        start_idx=200, 
        end_idx=len(raw_data),
        raw_data=raw_data
    )

    optimised_buy_spread = best_results_by_stock[ticker]['parameters']['buy_spread']
    optimised_sell_spread = best_results_by_stock[ticker]['parameters']['sell_spread']
    optimised_adx_threshold = best_results_by_stock[ticker]['parameters']['adx_cutoff']
    optimised_stop_loss = best_results_by_stock[ticker]['parameters']['stop_loss']
    optimised_profit_target = best_results_by_stock[ticker]['parameters']['profit_target']

    def calculate_adx(high, low, close, period=14):
        """Calculates the Average Directional Index (ADX) preserving Pandas tracking"""
        # 1. Calculate True Range (TR)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 2. Calculate Directional Movement (+DM and -DM)
        up_move = high.diff()
        down_move = -low.diff()
        
        # Using pandas .where instead of numpy preserves lengths and indexes
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        
        # 3. Smooth TR, +DM, and -DM using Wilder's technique
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        pos_di = 100 * pos_dm.ewm(alpha=1/period, adjust=False).mean() / atr
        neg_di = 100 * neg_dm.ewm(alpha=1/period, adjust=False).mean() / atr
        
        # 4. Calculate Directional Index (DX) and Average Directional Index (ADX)
        dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx

    high_prices = raw_data['High'][ticker]
    low_prices = raw_data['Low'][ticker]
    close_prices_ticker = close_prices[ticker]

    adx_values = calculate_adx(high_prices, low_prices, close_prices_ticker, period=14)

    # Allows for the calling of columns without the MultiIndex tuple if present
    if isinstance(close_prices.columns, pd.MultiIndex):
        close_prices.columns = [col[1] for col in close_prices.columns]
        volumes.columns = [col[1] for col in volumes.columns]

    # Calculates change in price for each ticker and the corresponding gains and losses
    price_changes = close_prices.diff()
    gains = price_changes.clip(lower=0)
    losses = -price_changes.clip(upper=0)

    # Calculate a short and a longer EMA
    ema_period_26 = 26
    ema_values_26 = raw_data['Close'][ticker].ewm(span=ema_period_26, adjust=False).mean()
    ema_period_12 = 12
    ema_values_12 = raw_data['Close'][ticker].ewm(span=ema_period_12, adjust=False).mean()
    ema_period_200 = 200
    ema_values_200 = raw_data['Close'][ticker].ewm(span=ema_period_200, adjust=False).mean()
    ema_values_50 = raw_data['Close'][ticker].ewm(span=50, adjust=False).mean()

    # Calculating EMA crossover signals
    ema_values_12_changes = ema_values_12.diff()
    ema_gains_12 = ema_values_12_changes.clip(lower=0)
    ema_losses_12 = -ema_values_12_changes.clip(upper=0)

    ema_values_26_changes = ema_values_26.diff()
    ema_gains_26 = ema_values_26_changes.clip(lower=0)
    ema_losses_26 = -ema_values_26_changes.clip(upper=0)

    # Simulating a simple trading strategy based on EMA crossovers
    cash = 10000.0
    position = 0               
    purchase_price = 0.0      
    portfolio_value = [] 
    trailing_floor_history = []  

    # Loop using an index 
    for i in range(len(close_prices)):
        close_price = close_prices[ticker].iloc[i]
        adx_today = adx_values.iloc[i]

        # Since i = 0 the first iteration is skipped
        if i < 200:
            portfolio_value.append(cash)
            continue

        # Calculate the signals inside the loop for day 'i'
        e12_today = ema_values_12.iloc[i]
        e26_today = ema_values_26.iloc[i]
        e12_yest = ema_values_12.iloc[i-1]
        e26_yest = ema_values_26.iloc[i-1]
        e200_today = ema_values_200.iloc[i]
        e200_yest = ema_values_200.iloc[i-1]
        e50_today = ema_values_50.iloc[i]
        e50_yest = ema_values_50.iloc[i-1]

        ema_spread_12_26 = (e12_today - e26_today) / e26_today

        ema_buy_signal = (ema_spread_12_26 > optimised_buy_spread) #and (e50_today > e200_today)
        ema_sell_signal = (ema_spread_12_26 < optimised_sell_spread) 

        # Execution Logic
        if ema_buy_signal and adx_today >= optimised_adx_threshold and position == 0:
            position = int(cash // close_price)
            purchase_price = close_price
            max_price = close_price
            cash -= position * close_price
        #print(f"Bought {position} shares at {close_price:.2f}") 

        elif position > 0:
            if close_price > max_price:
                max_price = close_price  

            trailing_floor = max_price * (optimised_stop_loss)
            if close_price < trailing_floor:
                cash += position * close_price
                #print(f"Trailing stop triggered. Sold at {close_price:.2f} | Peak was {max_price:.2f}")
                position = 0
                purchase_price = 0.0
                

            elif close_price >= purchase_price * optimised_profit_target: # Sell at an 8% gain
                cash += position * close_price
                position = 0
                purchase_price = 0.0

            elif ema_sell_signal and adx_today >= optimised_adx_threshold and position > 0:
                cash += position * close_price
                #print(f"Sold {position} shares at {close_price:.2f} | Profit: {(close_price - purchase_price) * position:.2f}")
                position = 0
                purchase_price = 0.0

            #elif current_time.hour == 15:
                #cash += position * close_price
                #print(f"Market closing. Sold {position} shares at {close_price:.2f} | Profit/Loss: {(close_price - purchase_price) * position:.2f}")
                #position = 0
                #purchase_price = 0.0
            
        # Track portfolio value at the end of every single step
        current_step_value = float(cash + (position * close_price))
        portfolio_value.append(current_step_value)
        
        if position > 0:
            trailing_floor_history.append(max_price * (optimised_stop_loss))
        else:
            trailing_floor_history.append(np.nan)

    # Find the starting values at index 200 where the trading loop begins
    stock_start_price = close_prices[ticker].iloc[200]
    portfolio_start_value = portfolio_value[200]

    # Convert both datasets into percentages starting at 100%
    normalized_stock = (close_prices[ticker] / stock_start_price) * 100
    normalized_portfolio = (np.array(portfolio_value) / portfolio_start_value) * 100
    normalized_sp500 = (close_prices["SPY"] / close_prices["SPY"].iloc[200]) * 100

    # Print results summary
    portfolio_return = normalized_portfolio[-1] - 100
    stock_return = normalized_stock.iloc[-1] - 100
    print(f"\nFinal Portfolio Value: ${portfolio_value[-1]:.2f} | Total Return: {portfolio_return:.2f}%")
    print(f"{ticker} Stock Performance: {stock_return:.2f}% over the same period.")

    # Create the chart
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot the percentage returns of the portfolio against the stock's performance
    axs[0].plot(close_prices.index[200:], normalized_stock[200:], label=f'{ticker} Performance (%)', alpha=0.5)
    axs[0].plot(close_prices.index[200:], normalized_portfolio[200:], label='Strategy Performance (%)', color='orange', linewidth=2)
    axs[1].plot(close_prices.index[200:], ema_values_12[200:], label='EMA 12', color='green', linestyle='--', alpha=0.7)
    axs[1].plot(close_prices.index[200:], ema_values_26[200:], label='EMA 26', color='red', linestyle='--', alpha=0.7)
    axs[1].plot(close_prices.index[200:], trailing_floor_history, label=f'Trailing Stop ({int(optimised_stop_loss * 100)}% of Peak)', color='blue', linestyle=':', alpha=0.7)
    axs[0].plot(close_prices.index[200:], normalized_sp500[200:], label='SPY Performance (%)', color='gray', linestyle='-', alpha=0.5)
    axs[1].plot(close_prices.index[200:], ema_values_200[200:], label='EMA 200', color='purple', linestyle='-', alpha=0.5)
    axs[1].plot(close_prices.index[200:], ema_values_50[200:], label='EMA 50', color='brown', linestyle='-', alpha=0.5)
    axs[0].set_xlabel('Date/Time')
    axs[0].set_ylabel('Growth / Return (%)')
    axs[0].set_title(f'{ticker} vs Strategy Performance (Normalized to 100%)')
    axs[0].legend()
    axs[1].set_xlabel('Date/Time')
    axs[1].set_ylabel('EMA Values')
    axs[1].set_title('Exponential Moving Averages')
    axs[1].legend()
    axs[0].grid()
    axs[1].grid()
    plt.show()