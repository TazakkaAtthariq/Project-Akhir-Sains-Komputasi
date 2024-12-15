import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Memuat file CSV yang diunggah
df = pd.read_csv('Data_Tesla.csv')

# Mengintegrasikan perubahan
predict_days = 10
interval = 1
method = 'linear'
# method = 'quadratic'
# method = 'cubic'

# Mengubah data menjadi bentuk yang sesuai
df['Date'] = pd.to_datetime(df['Date'])
df['Days'] = (df['Date'] - df['Date'].min()).dt.days + 1
df['Change %'] = df['Change %'].str.replace('%', '').astype(float) / 100

if df['Vol.'].str.contains('M').any():
    df['Vol.'] = df['Vol.'].str.replace('M', '').astype(float)
elif df['Vol.'].str.contains('K').any():
    df['Vol.'] = df['Vol.'].str.replace('K', '').astype(float)

# Interpolasi Data
def interpolate_data(df, columns_to_interpolate, method = method):
    interpolated_data = {}
    x = df['Days'].values

    for col in columns_to_interpolate:
        y = df[col].values
        interp_func = interp1d(x, y, kind=method, bounds_error=False, fill_value='extrapolate')

        x_interp = np.arange(x.min(), x.max() + 1)
        y_interp = interp_func(x_interp)

        interpolated_data[col] = np.round(y_interp, 2)
    interpolated_df = pd.DataFrame(interpolated_data)

    interpolated_df['Days'] = x_interp
    interpolated_df['Date'] = pd.to_datetime(df['Date'].min()) + pd.to_timedelta(interpolated_df['Days'] - 1, unit='D')
    interpolated_df = interpolated_df[['Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']]

    return interpolated_df

# Monte Carlo
def monte_carlo(df, predict_days, interval, simulations=10):
    # Mengambil data perubahan harian dan harga terakhir
    daily_changes = df['Change %']
    last_price = df['Price'].iloc[-1]

    # Menghitung rata-rata perubahan harian
    total_changes = 0
    for change in daily_changes:
        total_changes += change
    mean_change = total_changes / len(daily_changes)

    # Menghitung deviasi standar perubahan harian
    variance = 0
    for change in daily_changes:
        variance += (change - mean_change) ** 2
    std_dev_change = (variance / len(daily_changes)) ** 0.5

    future_prices = []
    future_intervals = np.arange(df['Days'].max() + interval, df['Days'].max() + predict_days + 1, interval)

    for future_day in future_intervals - df['Days'].max():
        simulated_prices = []
        for _ in range(simulations):
            simulated_price = last_price
            for _ in range(future_day):
                # Menghasilkan perubahan acak dari distribusi normal
                change = np.random.normal(mean_change, std_dev_change)
                simulated_price *= (1 + change)
            simulated_prices.append(simulated_price)
        
        # Menghitung rata-rata harga dari simulasi untuk interval saat ini
        total_price = 0
        for price in simulated_prices:
            total_price += price
        average_price = total_price / simulations

        future_prices.append(average_price)

    return future_intervals, future_prices

# Markov Chain
def markov_chain(df, predict_days, interval):
    # Mengambil perubahan harian dan mendefinisikan bin untuk state Markov
    changes = df['Change %']
    min_change = changes.min()
    max_change = changes.max()
    step = (max_change - min_change) / 5
    bins = [min_change + i * step for i in range(6)]

    # Menentukan state berdasarkan bin
    states = []
    for change in changes:
        for i in range(len(bins) - 1):
            if bins[i] <= change <= bins[i + 1]:
                states.append(i)
                break

    # Membentuk matriks transisi
    transition_matrix = [[0 for _ in range(len(bins) - 1)] for _ in range(len(bins) - 1)]
    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]
        transition_matrix[current_state][next_state] += 1
    
    # Menormalkan matriks transisi
    for i in range(len(transition_matrix)):
        total_transitions = sum(transition_matrix[i])
        if total_transitions > 0:
            for j in range(len(transition_matrix[i])):
                transition_matrix[i][j] /= total_transitions

    # Memulai prediksi harga di masa depan
    last_price = df['Price'].iloc[-1]
    last_state = states[-1]

    future_prices = []
    future_intervals = list(range(df['Days'].max() + interval, df['Days'].max() + predict_days + 1, interval))
    print(future_intervals)
    for future_day in future_intervals - df['Days'].max():
        current_price = last_price
        current_state = last_state
        
        for _ in range(future_day):
            probabilities = transition_matrix[current_state]
            next_state = np.random.choice(range(len(bins) - 1), p=probabilities)
            change = (bins[next_state] + bins[next_state + 1]) / 2
            current_price *= (1 + change)
            current_state = next_state

        future_prices.append(current_price)

    return future_intervals, future_prices

# Interpolasi
columns_to_interpolate = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
interpolated_df = interpolate_data(df, columns_to_interpolate, method)
interpolated_df.to_csv('Interpolated_Data.csv', index=False)
print(interpolated_df)

# Prediksi dengan Monte Carlo
future_days_Mcarlo, predicted_prices_Mcarlo = monte_carlo(df, predict_days, interval)
future_dates_Mcarlo = df['Date'].max() + pd.to_timedelta(future_days_Mcarlo - df['Days'].max(), unit='d')

# Prediksi dengan Markov Chain
future_days_Mchain, predicted_prices_Mchain = markov_chain(df, predict_days, interval)
future_dates_Mchain = df['Date'].max() + pd.to_timedelta(future_days_Mchain - df['Days'].max(), unit='d')

# Tampilkan hasil
print("\nPrediksi Monte Carlo:")
for date, price in zip(future_dates_Mcarlo, predicted_prices_Mcarlo):
    print(f"Tanggal: {date.date()}, Harga: {price:.2f}")

print("\nPrediksi Markov Chain:")
for date, price in zip(future_dates_Mchain, predicted_prices_Mchain):
    print(f"Tanggal: {date.date()}, Harga: {price:.2f}")

# Visualisasi hasil
plt.figure(figsize=(12, 6))
plt.plot(interpolated_df['Date'], interpolated_df['Price'], marker='o', linestyle='--', color='red', label='Interpolasi')
plt.plot(df['Date'], df['Price'], marker='o', linestyle='-', color='blue', label='Harga Aktual')
plt.scatter(future_dates_Mcarlo, predicted_prices_Mcarlo, color='orange', label='Prediksi Monte Carlo')
plt.scatter(future_dates_Mchain, predicted_prices_Mchain, color='green', label='Prediksi Markov Chain')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.title('Prediksi Harga')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()