import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import time
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="BTC ML Dashboard", layout="wide")
st_autorefresh(interval=60000, key="auto_refresh")

st.title("🚀 BTC-USD AI Trading Dashboard  💰💰")
col1, col2 = st.columns([8, 1])

with col2:
    if st.button("🔄 Refresh Data"):
        with st.spinner("Updating Market Data..."):
            time.sleep(1)
            st.rerun()
            
st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")            

# ----------------------------------------
# Download Data
# ----------------------------------------
df = yf.download("BTC-USD", period="2y", interval="1d", progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.dropna(inplace=True)

# ----------------------------------------
# Feature Engineering (same as your model)
# ----------------------------------------
close = df["Close"]

df["return_1d"]  = close.pct_change(1)
df["return_7d"]  = close.pct_change(7)
df["RSI"]        = RSIIndicator(close=close).rsi()
df["MACD_diff"]  = MACD(close=close).macd_diff()
bb = BollingerBands(close=close)
df["BB_position"] = (close - bb.bollinger_lband()) / (
    bb.bollinger_hband() - bb.bollinger_lband() + 1e-9
)

df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

df.dropna(inplace=True)

features = ["return_1d", "return_7d", "RSI", "MACD_diff", "BB_position"]

X = df[features]
y = df["Target"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ----------------------------------------
# Train Model
# ----------------------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_sc, y_train)

# ----------------------------------------
# Latest Prediction
# ----------------------------------------
latest_features = scaler.transform([X.iloc[-1]])
prediction = model.predict(latest_features)[0]
probability = model.predict_proba(latest_features)[0][1]

signal = "🟢 BUY" if prediction == 1 else "🔴 SELL"

# ----------------------------------------
# Metrics Cards
# ----------------------------------------
latest = df.iloc[-1]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Current Price",
    f"${latest['Close']:.2f}"
)

return7 = latest["return_7d"] * 100
col2.metric(
    "7D Return",
    f"{return7:.2f}%",
    delta=f"{return7:.2f}%"
)

col3.metric(
    "RSI",
    f"{latest['RSI']:.1f}"
)

col4.metric(
    "Model Signal",
    signal
)

st.markdown(f"### 🧠 Model Confidence: {probability*100:.2f}%")

# ----------------------------------------
# Live Price Chart
# ----------------------------------------
st.subheader("📈 BTC Price Chart")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Close"],
    mode="lines",
    name="BTC Price"
))

fig.update_layout(
    height=400,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# Strategy Backtest
# ----------------------------------------
st.subheader("📊 Strategy Performance")

y_pred = model.predict(X_test_sc)

test_df = df.iloc[split:].copy()
test_df["Predicted"] = y_pred

test_df["strategy_return"] = np.where(
    test_df["Predicted"] == 1,
    test_df["Close"].pct_change(),
    0
)

test_df["buy_hold_return"] = test_df["Close"].pct_change()

test_df["cum_strategy"] = (1 + test_df["strategy_return"]).cumprod()
test_df["cum_buy_hold"] = (1 + test_df["buy_hold_return"]).cumprod()

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=test_df.index,
    y=test_df["cum_strategy"],
    mode="lines",
    name="AI Strategy"
))

fig2.add_trace(go.Scatter(
    x=test_df.index,
    y=test_df["cum_buy_hold"],
    mode="lines",
    name="Buy & Hold"
))

fig2.update_layout(
    height=400,
    template="plotly_dark"
)

st.plotly_chart(fig2, use_container_width=True)

st.success("Dashboard Loaded Successfully 🚀")
st.info("This is a demo dashboard for educational purposes. Always do your own research before trading!.created by @HansakaV")