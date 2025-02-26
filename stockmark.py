

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import streamlit as st

# Streamlit UI setup
st.set_page_config(page_title="Stock Market Predictor", layout="wide", page_icon="📊")

# Custom CSS for modern UI
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .main-container {
        background-color: rgba(0, 0, 0, 0.85);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .title {
        text-align: center;
        font-size: 3.5em;
        color: black;
        margin-bottom: 25px;
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .input-box {
        margin: 20px 0;
    }
    .info-message {
        color: #FFD700;
        font-size: 1.3em;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<div class='title'>📊 Stock Market Predictor</div>", unsafe_allow_html=True)

# User input for stock tickers
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
search_input = st.text_input("Enter stock tickers separated by commas (e.g., TSLA, AAPL, GOOGL):", key='ticker_input')

# Process user input
tickers = [ticker.strip().upper() for ticker in search_input.split(',')] if search_input.strip() else []

if not tickers:
    st.markdown("<div class='info-message'>🔍 Please enter stock tickers to get started.</div>", unsafe_allow_html=True)
else:
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Fetch stock data
    st.write("Fetching stock data...")
    try:
        df = yf.download(tickers, start='2020-01-01', end=current_date, group_by='ticker')
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    # Display the latest stock data
    st.write("### Latest Stock Data")
    for ticker in tickers:
        if ticker in df.columns.levels[0]:
            st.write(f"**{ticker}**")
            st.write(df[ticker].tail())
        else:
            st.warning(f"Data not found for {ticker}.")

    # Prediction and Visualization
    for ticker in tickers:
        try:
            st.write(f"## {ticker} Stock Analysis")

            # Handle MultiIndex DataFrame
            if ticker in df.columns.levels[0]:
                if 'Close' in df[ticker].columns:
                    stock_data = df[ticker]['Close'].dropna().reset_index()
                else:
                    st.warning(f"'Close' data not found for {ticker}.")
                    continue
            elif 'Close' in df.columns:
                stock_data = df['Close'].dropna().reset_index()
            else:
                st.warning(f"'Close' data not found for {ticker}.")
                continue

            # Convert 'Date' to datetime and ordinal
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data['Date_Ordinal'] = stock_data['Date'].map(datetime.toordinal)

            # Prepare data for Linear Regression
            X = stock_data[['Date_Ordinal']]
            y = stock_data['Close']

            # Check for sufficient data
            if len(stock_data) < 2:
                st.warning(f"Not enough data to predict for {ticker}.")
                continue

            # Train Linear Regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predict next 5 days
            future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 6)]
            future_ordinals = np.array([[date.toordinal()] for date in future_dates])
            predicted_prices = model.predict(future_ordinals)

            # Display predictions
            prediction_table = pd.DataFrame({
                'Date': [date.strftime('%Y-%m-%d') for date in future_dates],
                'Predicted Price': [f"${price:.2f}" for price in predicted_prices]
            })

            st.write(f"**Predicted closing prices for the next 5 days for {ticker}:**")
            st.table(prediction_table)

            # Plot
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(stock_data['Date'], stock_data['Close'], label='Historical Prices', color='#FFD700')
            ax.scatter(future_dates, predicted_prices, color='#FF4500', label='Predicted Prices', zorder=5)
            ax.set_title(f"{ticker} Stock Price Prediction", color='white', fontsize=18)
            ax.set_xlabel('Date', color='white')
            ax.set_ylabel('Closing Price', color='white')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            fig.patch.set_facecolor('#1f1c2c')
            ax.set_facecolor('#2c2f33')
            ax.tick_params(colors='white')

            # Display plot
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred while processing {ticker}: {e}")

st.markdown("</div>", unsafe_allow_html=True)