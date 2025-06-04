# 📈 Stock Trend Predictor

This project is a web-based stock price prediction application using **LSTM (Long Short-Term Memory)** neural networks and **Streamlit**. It uses historical stock data from Yahoo Finance to visualize trends and predict future closing prices for any given stock symbol (e.g., GOOG, AAPL).

---

## 🚀 Features

- 📉 Historical stock data visualization
- 📊 Moving Averages (100, 200, 250 days)
- 🤖 LSTM-based time series prediction
- 📈 Graph comparison: Actual vs Predicted Prices
- 🌐 Streamlit-powered interactive web app
- 🔌 Publicly accessible via **Ngrok**

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**, **NumPy**, **Matplotlib**
- **Keras (TensorFlow backend)**
- **Scikit-learn**
- **YFinance** – for real-time data fetching
- **Streamlit** – for the frontend app
- **Ngrok** – to deploy locally and share publicly

---

## 🧠 Model Architecture

- LSTM layer (128 units) → LSTM layer (64 units) → Dense (25) → Dense (1)
- Trained on last 20 years of stock data
- Uses MinMaxScaler for normalization
- Predicts based on 100-day input windows

---

## 📂 Project Structure

📁 Stock-Trend-Predictor/
│
├── 📜 app.py # Streamlit frontend
├── 📓 Stock_Prediction.ipynb # Training and modeling notebook
├── 📦 Latest_stock_price_model.keras # Saved LSTM model
├── 📄 README.md # Project documentation (this file)

---

## 🔧 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/stock-trend-predictor.git
cd stock-trend-predictor

2. Install required packages
pip install -r requirements.txt

If you don't have a requirements.txt, use:
pip install streamlit yfinance keras scikit-learn matplotlib pandas numpy pyngrok

3. Run the app locally
streamlit run app.py

4. (Optional) Expose your app using Ngrok
from pyngrok import ngrok
ngrok.set_auth_token('your_ngrok_token')
ngrok.connect(8501)
🧪 How to Use
Run the app using Streamlit.

Enter a stock symbol (e.g., GOOG, AAPL, TSLA).

View historical charts, moving averages, and prediction graphs.

Compare actual vs predicted closing prices.

📷 Screenshots





📌 Future Improvements
Allow selection of time intervals (1Y, 5Y, Max)

Add more model types (GRU, ARIMA)

Deploy on cloud (Heroku, AWS, etc.)

Add technical indicators like RSI, MACD

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

📬 Contact
Made by Sujeet Pal
email : sujeetpal2606@gmail.com

