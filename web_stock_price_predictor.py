import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("Stock Price Predictor App")

# Get the stock symbol input from the user
stock = st.text_input("Enter the Stock ID", "GOOG")

# Set the date range for 20 years back from today
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# Download the stock data from Yahoo Finance
google_data = yf.download(stock, start, end)

# Load the pre-trained model
model = load_model("Latest_stock_price_model.keras")

# Display the stock data
st.subheader("Stock Data")
st.write(google_data)

# Split data into training and testing sets (70% training, 30% testing)
splitting_len = int(len(google_data) * 0.7)

# Access the 'Close' column correctly (multi-index)
x_test = pd.DataFrame(google_data[('Close', stock)][splitting_len:])

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data[('Close', stock)], 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Plot moving averages
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data[('Close', stock)].rolling(250).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data, 0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data[('Close', stock)].rolling(200).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data[('Close', stock)].rolling(100).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the 'Close' data (correct reference with multi-index)
scaled_data = scaler.fit_transform(x_test.values.reshape(-1, 1))

# Prepare the data for model input (use the last 100 values to predict the next one)
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

# Convert to numpy arrays
x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions using the trained model
predictions = model.predict(x_data)

# Inverse transform the scaled predictions and actual values to get the original price scale
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Create a DataFrame to display original vs predicted values
ploting_data = pd.DataFrame({
    'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
}, index=google_data.index[splitting_len+100:])

# Display the original vs predicted values
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Plot the original vs predicted stock prices
st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data[('Close', stock)][:splitting_len+100], ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)
