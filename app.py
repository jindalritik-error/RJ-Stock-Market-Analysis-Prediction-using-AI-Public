
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="RJ | Stock Market AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------
# PREMIUM CUSTOM DESIGN
# --------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #E5E7EB;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263247;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #F8FAFC;
}

.sub-title {
    color: #94A3B8;
    font-size: 14px;
}

.metric-card {
    background: linear-gradient(135deg, #172337, #111827);
    border: 1px solid #263750;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}

.metric-label {
    color: #94A3B8;
    font-size: 13px;
}

.metric-value {
    color: #F8FAFC;
    font-size: 27px;
    font-weight: 700;
}

.section-title {
    color: #EABF67;
    font-size: 22px;
    font-weight: 700;
    margin-top: 20px;
}

div.stButton > button {
    background-color: #D6A84F;
    color: #111827;
    border-radius: 8px;
    border: none;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #F0C875;
    color: #111827;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# HISTORICAL STOCK DATA
# --------------------------------

@st.cache_data(ttl=900)
def get_stock_data(stock, period):

    data = yf.download(
        stock,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data.columns.name = None

    return data


# --------------------------------
# INTRADAY LIVE DATA
# --------------------------------

@st.cache_data(ttl=60)
def get_intraday_data(stock):

    data = yf.download(
        stock,
        period="1d",
        interval="5m",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data.columns.name = None

    return data


# --------------------------------
# LIVE MARKET PANEL
# --------------------------------

@st.fragment(run_every="5m")
def show_live_market(stock, company):

    st.markdown(
        '<div class="section-title">LIVE MARKET MONITOR</div>',
        unsafe_allow_html=True
    )

    live_data = get_intraday_data(stock)

    if live_data.empty:
        st.warning(
            "Intraday data is currently unavailable. "
            "Please try again later."
        )
        return

    latest = live_data.iloc[-1]
    current = float(latest["Close"])

    if len(live_data) > 1:

        previous = float(live_data.iloc[-2]["Close"])

        change = current - previous

        change_percent = (
            change / previous
        ) * 100 if previous != 0 else 0

    else:
        change = 0
        change_percent = 0

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Latest 5-Minute Price",
        f"₹{current:,.2f}",
        f"{change:+.2f}"
    )

    c2.metric(
        "5-Minute Change",
        f"{change_percent:+.2f}%"
    )

    c3.metric(
        "Latest Volume",
        f"{int(latest['Volume']):,}"
    )

    time_column = (
        "Datetime" if "Datetime" in live_data.columns
        else "Date" if "Date" in live_data.columns
        else live_data.columns[0]
    )

    live_fig = go.Figure()

    live_fig.add_trace(
        go.Scatter(
            x=live_data[time_column],
            y=live_data["Close"],
            name="5-Minute Price",
            line=dict(color="#22C55E", width=2)
        )
    )

    live_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#111827",
        height=300,
        xaxis_title="Time",
        yaxis_title="Price (INR)",
        hovermode="x unified"
    )

    st.plotly_chart(
        live_fig,
        use_container_width=True
    )

    st.caption(
        "Automatically refreshes every 5 minutes while "
        "the app session is active. Data timing depends "
        "on Yahoo Finance availability."
    )


# --------------------------------
# FEATURE ENGINEERING
# --------------------------------

def prepare_data(data):

    data = data.copy()

    data["MA_50"] = data["Close"].rolling(50).mean()

    data["MA_100"] = data["Close"].rolling(100).mean()

    data["Daily_Return"] = data["Close"].pct_change() * 100

    data["Next_Close"] = data["Close"].shift(-1)

    return data


# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.markdown("## 📈 RJ STOCK AI")

st.sidebar.markdown("---")

st.sidebar.markdown("### Market Settings")

stock_options = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "Tata Motors": "TATAMOTORS.NS"
}

selected_company = st.sidebar.selectbox(
    "Select Company",
    list(stock_options.keys())
)

stock = stock_options[selected_company]

period = st.sidebar.selectbox(
    "Historical Period",
    ["1y", "2y", "5y"],
    index=1
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "RJ Stock Market AI | Built with Python & Machine Learning"
)

analyze = st.sidebar.button("Analyze Market")


# --------------------------------
# MAIN HEADER
# --------------------------------

st.markdown(
    '<div class="main-title">Stock Market Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI-powered market analytics and experimental price forecasting</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# --------------------------------
# LIVE MARKET MONITOR
# --------------------------------

show_live_market(stock, selected_company)

st.markdown("---")


# --------------------------------
# ANALYSIS
# --------------------------------

if analyze:

    with st.spinner("Fetching market data..."):

        raw_data = get_stock_data(stock, period)

    if raw_data.empty:

        st.error(
            "Unable to fetch stock data. Please try again later."
        )

    else:

        required_columns = [
            "Open", "High", "Low", "Close", "Volume"
        ]

        if not all(
            col in raw_data.columns
            for col in required_columns
        ):

            st.error("Required stock data columns are missing.")

        else:

            df = prepare_data(raw_data)

            features = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "MA_50",
                "MA_100",
                "Daily_Return"
            ]

            supervised = df.dropna(
                subset=features + ["Next_Close"]
            ).copy()

            latest_features = df.dropna(
                subset=features
            )[features].tail(1)

            if len(supervised) < 100 or latest_features.empty:

                st.error(
                    "Not enough historical data for model training."
                )

            else:

                X = supervised[features]
                y = supervised["Next_Close"]

                split = int(len(supervised) * 0.8)

                X_train = X.iloc[:split]
                X_test = X.iloc[split:]

                y_train = y.iloc[:split]
                y_test = y.iloc[split:]

                # RANDOM FOREST MODEL

                model = RandomForestRegressor(
                    n_estimators=150,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                mae = mean_absolute_error(
                    y_test, predictions
                )

                rmse = np.sqrt(
                    mean_squared_error(y_test, predictions)
                )

                r2 = r2_score(y_test, predictions)

                # FINAL MODEL

                final_model = RandomForestRegressor(
                    n_estimators=150,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )

                final_model.fit(X, y)

                next_price = final_model.predict(
                    latest_features
                )[0]

                # MARKET OVERVIEW

                latest = df.iloc[-1]

                current_price = float(latest["Close"])

                previous_price = float(
                    df["Close"].iloc[-2]
                )

                price_change = current_price - previous_price

                change_percent = (
                    price_change / previous_price
                ) * 100

                st.markdown(
                    '<div class="section-title">MARKET OVERVIEW</div>',
                    unsafe_allow_html=True
                )

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric(
                        "Latest Price",
                        f"₹{current_price:,.2f}",
                        f"{price_change:+.2f}"
                    )

                with c2:
                    st.metric(
                        "Daily Change",
                        f"{change_percent:+.2f}%"
                    )

                with c3:
                    st.metric(
                        "52-Week High (Selected Period)",
                        f"₹{df['High'].max():,.2f}"
                    )

                with c4:
                    st.metric(
                        "52-Week Low (Selected Period)",
                        f"₹{df['Low'].min():,.2f}"
                    )

                # HISTORICAL PRICE CHART

                st.markdown(
                    '<div class="section-title">PRICE ANALYSIS</div>',
                    unsafe_allow_html=True
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=df["Close"],
                        name="Closing Price",
                        line=dict(
                            color="#EABF67",
                            width=2.5
                        )
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=df["MA_50"],
                        name="50 Day MA",
                        line=dict(
                            color="#22C55E",
                            width=1.5
                        )
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=df["MA_100"],
                        name="100 Day MA",
                        line=dict(
                            color="#38BDF8",
                            width=1.5
                        )
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#111827",
                    height=480,
                    xaxis_title="Date",
                    yaxis_title="Price (INR)",
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        y=1.1
                    ),
                    margin=dict(
                        l=20, r=20, t=40, b=20
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # VOLUME CHART

                st.markdown(
                    '<div class="section-title">TRADING VOLUME</div>',
                    unsafe_allow_html=True
                )

                volume_fig = go.Figure()

                volume_fig.add_trace(
                    go.Bar(
                        x=df["Date"],
                        y=df["Volume"],
                        name="Trading Volume",
                        marker_color="#38BDF8"
                    )
                )

                volume_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#111827",
                    height=300,
                    xaxis_title="Date",
                    yaxis_title="Volume"
                )

                st.plotly_chart(
                    volume_fig,
                    use_container_width=True
                )

                # MODEL PERFORMANCE

                st.markdown(
                    '<div class="section-title">MODEL PERFORMANCE</div>',
                    unsafe_allow_html=True
                )

                m1, m2, m3 = st.columns(3)

                m1.metric("MAE", f"₹{mae:.2f}")

                m2.metric("RMSE", f"₹{rmse:.2f}")

                m3.metric("R² Score", f"{r2:.4f}")

                st.caption(
                    "Metrics are calculated on the chronological "
                    "test set, not the training data."
                )

                # ACTUAL VS PREDICTED

                st.markdown(
                    '<div class="section-title">ACTUAL VS PREDICTED</div>',
                    unsafe_allow_html=True
                )

                comparison = pd.DataFrame({
                    "Date": supervised["Date"].iloc[split:],
                    "Actual Price": y_test.values,
                    "Predicted Price": predictions
                })

                compare_fig = go.Figure()

                compare_fig.add_trace(
                    go.Scatter(
                        x=comparison["Date"],
                        y=comparison["Actual Price"],
                        name="Actual",
                        line=dict(
                            color="#22C55E",
                            width=2
                        )
                    )
                )

                compare_fig.add_trace(
                    go.Scatter(
                        x=comparison["Date"],
                        y=comparison["Predicted Price"],
                        name="Predicted",
                        line=dict(
                            color="#EABF67",
                            width=2
                        )
                    )
                )

                compare_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#111827",
                    height=400,
                    hovermode="x unified"
                )

                st.plotly_chart(
                    compare_fig,
                    use_container_width=True
                )

                # NEXT DAY ESTIMATE

                st.markdown(
                    '<div class="section-title">AI PRICE ESTIMATE</div>',
                    unsafe_allow_html=True
                )

                p1, p2 = st.columns(2)

                with p1:
                    st.metric(
                        "Experimental Next Trading Day Estimate",
                        f"₹{next_price:,.2f}"
                    )

                with p2:

                    estimate_change = (
                        (next_price - current_price)
                        / current_price
                    ) * 100

                    st.metric(
                        "Estimated Change",
                        f"{estimate_change:+.2f}%"
                    )

                st.warning(
                    "This is an experimental machine learning estimate, "
                    "not a guaranteed future price or investment recommendation."
                )

                # DOWNLOAD REPORT

                st.markdown(
                    '<div class="section-title">DOWNLOAD REPORT</div>',
                    unsafe_allow_html=True
                )

                csv_data = comparison.to_csv(index=False)

                st.download_button(
                    label="Download Prediction Report",
                    data=csv_data,
                    file_name=f"{stock}_prediction_report.csv",
                    mime="text/csv"
                )

                st.success(
                    "Market analysis completed successfully."
                )

                st.caption(
                    "Developed by Ritik Jindal | Stock Market AI Project"
                )


else:

    st.info(
        "Select a company and historical period from the sidebar, "
        "then click Analyze Market."
    )

    st.markdown("""
    ### Welcome to RJ Stock Market AI

    Explore historical stock prices, technical indicators,
    machine learning predictions and model evaluation.

    Select a stock from the sidebar to begin.
    """)
