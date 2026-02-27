import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow
import mlflow.sklearn

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# ─────────────────────────────────────
# 1. DATA DOWNLOAD
# ─────────────────────────────────────
df = yf.download('BTC-USD', period='5y', interval='1d', progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)

# ─────────────────────────────────────
# 2. TARGET (predict 1 day ahead)
# ─────────────────────────────────────
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# ─────────────────────────────────────
# 3. BETTER FEATURES (normalized — no raw price)
# ─────────────────────────────────────
close  = df["Close"].squeeze()
high   = df["High"].squeeze()
low    = df["Low"].squeeze()
volume = df["Volume"].squeeze()

# Returns (% change — scale independent)
df["return_1d"]  = close.pct_change(1)
df["return_3d"]  = close.pct_change(3)
df["return_7d"]  = close.pct_change(7)
df["return_14d"] = close.pct_change(14)
df["return_30d"] = close.pct_change(30)

# MA ratios (not raw MA values)
df["ma7_ma21_ratio"] = close.rolling(7).mean() / close.rolling(21).mean()
df["ma7_ma50_ratio"] = close.rolling(7).mean() / close.rolling(50).mean()
df["ma21_ma50_ratio"] = close.rolling(21).mean() / close.rolling(50).mean()

# Normalized volatility
df["volatility_7"]  = close.rolling(7).std() / close
df["volatility_21"] = close.rolling(21).std() / close

# RSI
df["RSI"] = RSIIndicator(close=close, window=14).rsi()
df["RSI_7"] = RSIIndicator(close=close, window=7).rsi()

# MACD difference (signal crossover)
macd = MACD(close=close)
df["MACD_diff"] = macd.macd_diff()

# Bollinger Band position (0 = at low band, 1 = at high band)
bb = BollingerBands(close=close, window=20)
df["BB_position"] = (close - bb.bollinger_lband()) / (
    bb.bollinger_hband() - bb.bollinger_lband() + 1e-9
)
df["BB_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close

# ATR normalized
df["ATR"] = AverageTrueRange(high=high, low=low, close=close).average_true_range() / close

# Volume features
df["volume_change"]   = volume.pct_change(1)
df["volume_ma_ratio"] = volume / volume.rolling(7).mean()

# Price position within day's range
df["day_range_position"] = (close - low) / (high - low + 1e-9)

df.dropna(inplace=True)
print("Dataset shape after features:", df.shape)
print("Target balance:\n", df['Target'].value_counts())

# ─────────────────────────────────────
# 4. SPLIT
# ─────────────────────────────────────
features = [
    "return_1d", "return_3d", "return_7d", "return_14d", "return_30d",
    "ma7_ma21_ratio", "ma7_ma50_ratio", "ma21_ma50_ratio",
    "volatility_7", "volatility_21",
    "RSI", "RSI_7", "MACD_diff",
    "BB_position", "BB_width", "ATR",
    "volume_change", "volume_ma_ratio",
    "day_range_position"
]

X = df[features]
y = df['Target']

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────
# 5. HYPERPARAMETER TUNING
#    Using TimeSeriesSplit (correct for time data)
# ─────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)

# --- Random Forest Tuning ---
print("\nTuning RandomForest...")
rf_params = {
    "n_estimators":  [100, 200, 300, 500],
    "max_depth":     [3, 4, 5, 6, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5]
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rf_params,
    n_iter=30,              # tries 30 random combinations
    cv=tscv,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rf_search.fit(X_train_sc, y_train)
best_rf = rf_search.best_estimator_
print("Best RF params:", rf_search.best_params_)

# --- XGBoost Tuning ---
print("\nTuning XGBoost...")
xgb_params = {
    "n_estimators":    [100, 200, 300, 500],
    "max_depth":       [3, 4, 5, 6],
    "learning_rate":   [0.01, 0.02, 0.05, 0.1],
    "subsample":       [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree":[0.6, 0.7, 0.8, 0.9],
    "min_child_weight":[1, 3, 5, 7],
    "gamma":           [0, 0.1, 0.2, 0.5]
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss"),
    param_distributions=xgb_params,
    n_iter=30,
    cv=tscv,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
    verbose=1
)
xgb_search.fit(X_train_sc, y_train)
best_xgb = xgb_search.best_estimator_
print("Best XGB params:", xgb_search.best_params_)

# ─────────────────────────────────────
# 6. EVALUATE + MLFLOW
# ─────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///crypto_mlflow.db")
mlflow.set_experiment("Crypto-Tuned-V3")

tuned_models = {
    "RandomForest_Tuned": (best_rf, rf_search.best_params_),
    "XGBoost_Tuned":      (best_xgb, xgb_search.best_params_)
}

for model_name, (model, best_params) in tuned_models.items():
    with mlflow.start_run(run_name=model_name):

        y_pred = model.predict(X_test_sc)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)

        print(f"\n{'='*35}")
        print(f"Model     : {model_name}")
        print(f"Accuracy  : {acc:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")

        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "accuracy": acc, "f1_score": f1,
            "precision": prec, "recall": rec
        })

        os.makedirs("artifacts", exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        plt.figure(figsize=(5, 4))
        disp.plot()
        plt.title(model_name)
        path = f"artifacts/{model_name}_cm.png"
        plt.savefig(path)
        plt.close()
        mlflow.log_artifact(path)
        mlflow.sklearn.log_model(model, name=model_name)

# ─────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────
importances = pd.Series(best_rf.feature_importances_, index=features)
importances.sort_values().plot(kind="barh", figsize=(9, 7), color="steelblue")
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("artifacts/feature_importance.png")
plt.show()

# ─────────────────────────────────────
# 8. BACKTEST
# ─────────────────────────────────────
y_pred_best = best_xgb.predict(X_test_sc)

test_df = X_test.copy()
test_df["Predicted"] = y_pred_best
test_df["Close"]     = df["Close"].iloc[split:].values

test_df["strategy_return"] = np.where(
    test_df["Predicted"] == 1,
    test_df["Close"].pct_change(), 0
)
test_df["buy_hold_return"] = test_df["Close"].pct_change()
test_df["cum_strategy"] = (1 + test_df["strategy_return"]).cumprod()
test_df["cum_buy_hold"] = (1 + test_df["buy_hold_return"]).cumprod()

plt.figure(figsize=(12, 5))
plt.plot(test_df["cum_strategy"].values, label="Model Strategy", color="green")
plt.plot(test_df["cum_buy_hold"].values, label="Buy & Hold", color="orange")
plt.title("Strategy vs Buy & Hold (Tuned Model)")
plt.legend()
plt.savefig("artifacts/backtest_tuned.png")
plt.show()

print("\nDone! Open MLflow:")
print('& "C:\\Program Files\\Python310\\python.exe" -m mlflow ui --backend-store-uri sqlite:///crypto_mlflow.db')

#Variables for latest data point
latest = df.iloc[-1]
print(f"RSI: {latest['RSI']:.1f}")
print(f"7D Return: {latest['return_7d']*100:.2f}%")
print(f"Volume Change: {latest['volume_change']*100:.2f}%")
print(f"Current Price: {latest['Close']:.2f}")