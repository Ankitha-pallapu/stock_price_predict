import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Title
st.title("Stock Price Predictor App")

# Input for stock symbol
stock = st.text_input("Enter the Stock ID", "GOOG")

# Define date range
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download stock data
google_data = yf.download(stock, start, end)

# Load your trained model
model = load_model("Latest_stock_price_model.keras")

# Show stock data
st.subheader("Stock Data")
st.write(google_data)

# Split data
splitting_len = int(len(google_data) * 0.7)
x_test = google_data[['Close']].iloc[splitting_len:]

# Plotting function
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange', label='Moving Average')
    plt.plot(full_data.Close, 'b', label='Close Price')
    if extra_data and extra_dataset is not None:
        plt.plot(extra_dataset, 'g', label='Extra MA')
    plt.legend()
    return fig

# Moving averages
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data))

st.subheader('Original Close Price with MA for 100 and 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)

# Prepare test sequences
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Prediction
predictions = model.predict(x_data)

# Inverse transform
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Create dataframe for plotting
ploting_data = pd.DataFrame({
    'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
}, index=google_data.index[splitting_len + 100:])

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Final Plot
st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(google_data.Close[:splitting_len + 100], label="Data - Not Used")
plt.plot(ploting_data['original_test_data'], label="Original Test Data")
plt.plot(ploting_data['predictions'], label="Predicted Test Data")
plt.legend()
st.pyplot(fig)
