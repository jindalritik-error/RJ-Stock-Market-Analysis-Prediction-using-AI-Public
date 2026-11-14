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
        st.warning("Intraday data is currently unavailable.")
        return

    latest = live_data.iloc[-1]
    current = float(latest["Close"])

    if len(live_data) > 1:
        previous = float(live_data.iloc[-2]["Close"])
        change = current - previous
        change_percent = (change / previous) * 100
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
        "Refreshes every 5 minutes while this app session is active. "
        "Data availability and timing depend on Yahoo Finance."
    )
