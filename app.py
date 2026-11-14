
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Stock Market AI",
    page_icon="📈",
    layout="wide"
)

st.title("Stock Market Analysis & Prediction")
st.write("Machine Learning Based Stock Market Dashboard")

st.sidebar.header("Stock Selection")

stock = st.sidebar.selectbox(
    "Choose a Stock",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS",
     "HDFCBANK.NS", "ICICIBANK.NS"]
)

period = st.sidebar.selectbox(
    "Select Time Period",
    ["1y", "2y", "5y"]
)

if st.sidebar.button("Analyze Stock"):

    with st.spinner("Downloading stock data..."):

        df = yf.download(
            stock,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

    if df.empty:
        st.error("Stock data could not be downloaded.")

    else:

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df = df.dropna().copy()

        df["MA_50"] = df["Close"].rolling(50).mean()
        df["MA_100"] = df["Close"].rolling(100).mean()
        df["Daily_Return"] = df["Close"].pct_change() * 100

        df["Next_Close"] = df["Close"].shift(-1)

        features = [
            "Open", "High", "Low", "Close",
            "Volume", "MA_50", "MA_100",
            "Daily_Return"
        ]

        training_data = df.dropna().copy()

        if len(training_data) < 100:
            st.error("Not enough historical data for analysis.")

        else:

            X = training_data[features]
            y = training_data["Next_Close"]

            split = int(len(training_data) * 0.8)

            X_train = X.iloc[:split]
            X_test = X.iloc[split:]

            y_train = y.iloc[:split]
            y_test = y.iloc[split:]

            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            st.header("Stock Overview")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Latest Closing Price",
                f"₹{df['Close'].iloc[-1]:,.2f}"
            )

            col2.metric(
                "Highest Price",
                f"₹{df['High'].max():,.2f}"
            )

            col3.metric(
                "Lowest Price",
                f"₹{df['Low'].min():,.2f}"
            )

            st.header("Historical Stock Price")

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(df["Date"], df["Close"], label="Closing Price")
            ax.plot(df["Date"], df["MA_50"], label="50 Day Average")
            ax.plot(df["Date"], df["MA_100"], label="100 Day Average")

            ax.set_xlabel("Date")
            ax.set_ylabel("Price (INR)")
            ax.legend()
            ax.grid()

            st.pyplot(fig)
            plt.close(fig)

            st.header("Model Performance")

            col1, col2, col3 = st.columns(3)

            col1.metric("MAE", f"₹{mae:.2f}")
            col2.metric("RMSE", f"₹{rmse:.2f}")
            col3.metric("R2 Score", f"{r2:.3f}")

            st.header("Actual vs Predicted Prices")

            result = pd.DataFrame({
                "Actual Price": y_test.values,
                "Predicted Price": predictions
            })

            st.line_chart(result)

            st.header("Next Trading Day Estimate")

            latest_data = df[features].iloc[[-1]]

            if latest_data.isnull().any().any():
                st.warning("Not enough data for the latest estimate.")

            else:
                next_price = model.predict(latest_data)[0]

                st.metric(
                    "Experimental Model Estimate",
                    f"₹{next_price:,.2f}"
                )

            st.success("Stock analysis completed!")

            st.caption(
                "Educational project only. Predictions are experimental "
                "and are not financial or investment advice."
            )

