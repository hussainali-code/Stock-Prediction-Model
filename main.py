import numpy as np
import pandas as pd
import matplotlib
import sys
import os

# Check for non-interactive plot generation argument
save_only = "--save-only" in sys.argv
if save_only:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Ensure results directory exists
os.makedirs("results", exist_ok=True)



df = pd.read_csv("Amazon.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

print("Data Types:\n")
print(df.dtypes)

print("\nDataset Shape:", df.shape)

print("\nMissing Values:\n")
print(df.isnull().sum())



# Closing Price
plt.figure(figsize=(14,6))
plt.plot(df["date"], df["close"])
plt.title("Amazon Closing Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.savefig("results/closing_price.png", dpi=150, bbox_inches="tight")
plt.show()

# Open, High, Low, Close
plt.figure(figsize=(14,6))
plt.plot(df["date"], df["open"], label="Open")
plt.plot(df["date"], df["high"], label="High")
plt.plot(df["date"], df["low"], label="Low")
plt.plot(df["date"], df["close"], label="Close")

plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Amazon Stock Prices")
plt.grid(True)
plt.savefig("results/stock_prices.png", dpi=150, bbox_inches="tight")
plt.show()

# Histograms
df.hist(figsize=(12,8))
plt.tight_layout()
plt.savefig("results/histograms.png", dpi=150, bbox_inches="tight")
plt.show()

# Correlation Matrix
corr = df.corr(numeric_only=True)

print("\nCorrelation Matrix:\n")
print(corr)

plt.figure(figsize=(8,6))
plt.imshow(corr)

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.colorbar()
plt.title("Correlation Matrix")
plt.savefig("results/correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(df["open"], df["close"])
plt.xlabel("Open Price")
plt.ylabel("Close Price")
plt.title("Open vs Close")
plt.grid(True)
plt.savefig("results/open_vs_close.png", dpi=150, bbox_inches="tight")
plt.show()



X = df[["open", "high", "low", "volume"]].values
y = df["close"].values

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

# Feature Scaling
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)

X = (X - mu) / sigma

# Initialize Parameters
m, n = X.shape

w = np.zeros(n)
b = 0



def predict(X, w, b):
    return np.dot(X, w) + b



def compute_cost(X, y, w, b):

    m = X.shape[0]

    predictions = predict(X, w, b)

    cost = (1/(2*m)) * np.sum((predictions - y)**2)

    return cost



def compute_gradient(X, y, w, b):

    m = X.shape[0]

    predictions = predict(X, w, b)

    error = predictions - y

    dj_dw = (1/m) * np.dot(X.T, error)

    dj_db = (1/m) * np.sum(error)

    return dj_dw, dj_db



def gradient_descent(X, y, w, b, alpha, iterations):

    J_history = []

    for i in range(iterations):

        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X, y, w, b)

        J_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i:4d} | Cost = {cost:.6f}")

    return w, b, J_history


alpha = 0.01
iterations = 2000

w, b, J_history = gradient_descent(
    X,
    y,
    w,
    b,
    alpha,
    iterations
)

print("\nTraining Finished")

print("\nWeights:")
print(w)

print("\nBias:")
print(b)



plt.figure(figsize=(8,5))
plt.plot(J_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent Convergence")
plt.grid(True)
plt.savefig("results/gradient_descent_convergence.png", dpi=150, bbox_inches="tight")
plt.show()



predictions = predict(X, w, b)

print("\nFirst 10 Predictions")
print(predictions[:10])

print("\nFirst 10 Actual Values")
print(y[:10])



mse = np.mean((predictions - y)**2)
print(f"\nMean Squared Error (MSE): {mse:.6f}")

# Calculate R2 (R-squared) value
ss_res = np.sum((predictions - y) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f"R-squared (R2) Value: {r2:.6f}")

plt.figure(figsize=(8,6))

plt.scatter(y, predictions, alpha=0.6, label="Predictions")

minimum = min(y.min(), predictions.min())
maximum = max(y.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    'r--',
    linewidth=2,
    label="Perfect Fit"
)

plt.xlabel("Actual Close Price")
plt.ylabel("Predicted Close Price")
plt.title(f"Actual vs Predicted (MSE: {mse:.4f}, R2: {r2:.4f})")
plt.legend()
plt.grid(True)
plt.savefig("results/actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()

# If in save-only mode, exit before running the interactive lookup
if save_only:
    print("\n[Info] Plots successfully generated and saved to the 'results/' directory. Exiting --save-only mode.")
    sys.exit(0)







# Create a date-only representation for easier matching
df_date_only = df["date"].dt.date
min_date = df_date_only.min()
max_date = df_date_only.max()

print(f"Dataset date range: {min_date} to {max_date}")

while True:
    print("\nSelect an option:")
    print("1. Look up historical date in dataset")
    print("2. Enter custom values to predict closing price")
    print("3. Quit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        user_input = input("\nEnter Date (YYYY-MM-DD): ").strip()
        try:
            target_date = pd.to_datetime(user_input).date()
            row = df[df_date_only == target_date]
            
            if row.empty:
                print(f"Date '{user_input}' not found in the dataset.")
                print("Please ensure the date falls on a trading day (no weekends/holidays) within the range.")
                continue
                
            val_open = row["open"].values[0]
            val_high = row["high"].values[0]
            val_low = row["low"].values[0]
            val_volume = row["volume"].values[0]
            actual_close = row["close"].values[0]
            
            print(f"\nFound historical features for {target_date}:")
            print(f" - Open Price:   {val_open:.4f}")
            print(f" - High Price:   {val_high:.4f}")
            print(f" - Low Price:    {val_low:.4f}")
            print(f" - Volume:       {val_volume:,.0f}")
            print(f" - Actual Close:  {actual_close:.4f}")
            
            # Scale features and predict
            new_data = np.array([[val_open, val_high, val_low, val_volume]])
            new_data_scaled = (new_data - mu) / sigma
            predicted_price = predict(new_data_scaled, w, b)
            
            print(f"--> Predicted Closing Price: {predicted_price[0]:.4f}")
            print(f"--> Difference (Actual - Predicted): {actual_close - predicted_price[0]:.4f}")
        except Exception as e:
            print(f"[Error] Invalid date format or date parse error. (Details: {e})")
            
    elif choice == '2':
        print("\nEnter custom values:")
        try:
            val_open = float(input(" - Open Price: "))
            val_high = float(input(" - High Price: "))
            val_low = float(input(" - Low Price: "))
            val_volume = float(input(" - Volume: "))
            
            # Scale features and predict
            new_data = np.array([[val_open, val_high, val_low, val_volume]])
            new_data_scaled = (new_data - mu) / sigma
            predicted_price = predict(new_data_scaled, w, b)
            
            print(f"\n--> Predicted Closing Price: {predicted_price[0]:.4f}")
        except ValueError:
            print("[Error] Please enter valid numbers.")
        except Exception as e:
            print(f"[Error] An unexpected error occurred. (Details: {e})")
            
    elif choice == '3':
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

