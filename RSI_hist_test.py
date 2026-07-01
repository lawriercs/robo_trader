import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. DOWNLOAD HISTORICAL DATA
# -------------------------------------------------------------------
ticker = "TSLA"
data = yf.download(ticker, start="2024-01-01", end="2026-01-01")
data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

# -------------------------------------------------------------------
# 2. CALCULATE STRICT RSI (14-Period)
# -------------------------------------------------------------------
change = data['Close'].diff()
gain = change.mask(change < 0, 0)
loss = -change.mask(change > 0, 0)

avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()

rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))
data = data.dropna().copy()

# -------------------------------------------------------------------
# 3. RUN SIMULATION (RSI + 5% Stop-Loss + 10% Fixed Take-Profit)
# -------------------------------------------------------------------
cash = 10000.00
position = 0               
purchase_price = 0.0      
portfolio_value = [] 

print(f"Starting Adjusted Algorithm for {ticker}...")
print(f"Rules: Buy RSI <= 30 | Sell RSI >= 70 OR +10% Gain | Stop-Loss -5%\n" + "-"*65)

for date, row in data.iterrows():
    current_price = float(row['Close'])
    rsi_value = float(row['RSI'])
    
    # 1. CHECK RISK EXITS FIRST (Stop-Loss or Fixed Take-Profit)
    if position > 0 and current_price <= (purchase_price * 0.95):
        cash += position * current_price
        print(f"{date.strftime('%Y-%m-%d')}: !!! STOP-LOSS !!! Sold at ${current_price:.2f} (-5%)")
        position = 0
        purchase_price = 0.0
        
    elif position > 0 and current_price >= (purchase_price * 1.10):
        cash += position * current_price
        print(f"{date.strftime('%Y-%m-%d')}: $$$ TARGET HIT $$$ Sold at ${current_price:.2f} (+10%)")
        position = 0
        purchase_price = 0.0

    # 2. BUY SIGNAL (Strict RSI filter)
    elif rsi_value <= 30 and position == 0:
        position = int(cash // current_price)
        cash -= position * current_price
        purchase_price = current_price  
        print(f"{date.strftime('%Y-%m-%d')}: BOUGHT {position} shares at ${current_price:.2f} (RSI: {rsi_value:.1f})")
        
    # 3. ALTERNATIVE SELL SIGNAL (RSI Overbought fallback)
    elif rsi_value >= 70 and position > 0:
        cash += position * current_price
        print(f"{date.strftime('%Y-%m-%d')}: SOLD (RSI Overbought) at ${current_price:.2f} (RSI: {rsi_value:.1f})")
        position = 0
        purchase_price = 0.0
        
    portfolio_value.append(cash + (position * current_price))

data['Portfolio_Value'] = portfolio_value

# -------------------------------------------------------------------
# 4. SHOW RE-OPTIMIZED RESULTS
# -------------------------------------------------------------------
final_value = portfolio_value[-1]
total_return = ((final_value - 10000.00) / 10000.00) * 100

print("-"*65)
print(f"Simulation Complete.")
print(f"Final Portfolio Value: ${final_value:.2f} | Total Return: {total_return:.2f}%")


plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Portfolio_Value'], label='RSI + 5% SL + 10% Target', color='darkorange', linewidth=2)
plt.title(f'RSI Strategy with Fixed Profit Targets on {ticker}')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()