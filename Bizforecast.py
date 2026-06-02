"""
=============================================================================
BizForecast: Small Business Sales Demand Forecasting System
Description:
    An AI-based sales demand forecasting system that predicts future product
    demand using regression, time-series, KNN, ANN, CNN, Decision Tree,
    and Random Forest models, with inventory level recommendations and an
    interactive visualization dashboard.
=============================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS & CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from datetime import datetime, timedelta
import os

# Scikit-learn models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance

# TensorFlow / Keras for ANN and CNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D,
                                      Flatten, LSTM, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# Statsmodels for time-series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Plot style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'text.color': '#c9d1d9',
    'grid.color': '#21262d',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
})

PALETTE = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657',
           '#79c0ff', '#56d364', '#ff7b72', '#bc8cff', '#ffb86c']

OUTPUT_DIR = "/content/bizforecast_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  BizForecast — Small Business Sales Demand Forecasting System")
print("=" * 70)
print(f"  Output directory : {os.path.abspath(OUTPUT_DIR)}")
print(f"  TensorFlow       : {tf.__version__}")
print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA GENERATION (Synthetic Small Business Sales Data)
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Generating synthetic small business sales dataset...")

def generate_sales_data(n_days: int = 730, n_products: int = 5) -> pd.DataFrame:
    """
    Generates realistic synthetic daily sales data for multiple products,
    incorporating:
      - Long-term upward trend
      - Weekly seasonality (weekends spike)
      - Annual seasonality (festive season peaks)
      - Random noise
      - Promotional effects
      - Holiday effects
    """
    np.random.seed(RANDOM_SEED)
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    products = [f"Product_{chr(65+i)}" for i in range(n_products)]

    records = []
    for product in products:
        base_sales = np.random.randint(50, 200)
        trend_coeff = np.random.uniform(0.01, 0.08)
        noise_std = np.random.uniform(5, 20)

        for i, date in enumerate(dates):
            # Trend
            trend = base_sales + trend_coeff * i

            # Weekly seasonality
            dow = date.weekday()
            weekly_factor = 1.3 if dow >= 5 else (0.85 if dow == 0 else 1.0)

            # Annual seasonality (festival peaks: Oct-Dec)
            doy = date.timetuple().tm_yday
            annual_factor = 1 + 0.4 * np.sin((doy / 365) * 2 * np.pi - np.pi / 2)

            # Promotions (random months)
            promo = 1.0
            if date.month in [3, 8, 11] and date.day <= 10:
                promo = np.random.uniform(1.2, 1.6)

            # Holiday effect
            holiday = 1.0
            if (date.month == 10 and date.day in [2]) or \
               (date.month == 1 and date.day == 26) or \
               (date.month == 8 and date.day == 15):
                holiday = np.random.uniform(1.4, 2.0)

            # Compose sales
            sales = (trend * weekly_factor * annual_factor * promo * holiday
                     + np.random.normal(0, noise_std))
            sales = max(0, round(sales, 2))

            records.append({
                'date': date,
                'product': product,
                'sales': sales,
                'price': round(np.random.uniform(50, 500), 2),
                'is_weekend': int(dow >= 5),
                'is_holiday': int(holiday > 1.0),
                'is_promo': int(promo > 1.0),
                'month': date.month,
                'day_of_week': dow,
                'day_of_year': doy,
                'week_of_year': date.isocalendar()[1],
                'quarter': (date.month - 1) // 3 + 1,
            })

    df = pd.DataFrame(records)
    df.sort_values(['product', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


df_raw = generate_sales_data(n_days=730, n_products=5)
logger.info(f"Dataset shape: {df_raw.shape}")
print(f"\n[DATA] Sample rows:\n{df_raw.head(10).to_string()}\n")
print(f"[DATA] Statistics:\n{df_raw['sales'].describe().round(2)}\n")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Running Exploratory Data Analysis...")

def run_eda(df: pd.DataFrame):
    """Full EDA with multiple plots."""
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle("BizForecast — Exploratory Data Analysis", fontsize=18,
                 color='#e6edf3', fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Total daily sales across all products
    ax1 = fig.add_subplot(gs[0, :2])
    daily = df.groupby('date')['sales'].sum().reset_index()
    ax1.plot(daily['date'], daily['sales'], color=PALETTE[0], lw=1.2, alpha=0.8)
    ax1.fill_between(daily['date'], daily['sales'], alpha=0.15, color=PALETTE[0])
    ax1.set_title("Total Daily Sales (All Products)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Sales")
    ax1.tick_params(axis='x', rotation=30)

    # 2. Sales distribution by product (boxplot)
    ax2 = fig.add_subplot(gs[0, 2])
    products = df['product'].unique()
    data_bp = [df[df['product'] == p]['sales'].values for p in products]
    bp = ax2.boxplot(data_bp, patch_artist=True, notch=True,
                     medianprops={'color': 'white', 'lw': 2})
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xticklabels(products, rotation=25)
    ax2.set_title("Sales Distribution by Product")
    ax2.set_ylabel("Sales")

    # 3. Weekly seasonality heatmap
    ax3 = fig.add_subplot(gs[1, :2])
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    pivot = df.groupby(['day_of_week', 'month'])['sales'].mean().unstack()
    pivot.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    sns.heatmap(pivot, ax=ax3, cmap='YlOrRd', linewidths=0.5,
                cbar_kws={'label': 'Avg Sales'}, annot=False)
    ax3.set_title("Avg Sales Heatmap (Day of Week × Month)")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Day of Week")

    # 4. Monthly sales bar chart
    ax4 = fig.add_subplot(gs[1, 2])
    monthly = df.groupby('month')['sales'].mean()
    bars = ax4.bar(monthly.index, monthly.values,
                   color=[PALETTE[i % len(PALETTE)] for i in range(12)], edgecolor='none')
    ax4.set_title("Avg Monthly Sales")
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Avg Sales")
    ax4.set_xticks(range(1, 13))

    # 5. Promo vs Non-Promo
    ax5 = fig.add_subplot(gs[2, 0])
    promo_avg = df.groupby('is_promo')['sales'].mean()
    ax5.bar(['No Promo', 'Promo'], promo_avg.values,
            color=[PALETTE[2], PALETTE[1]], edgecolor='none', alpha=0.85)
    ax5.set_title("Sales: Promo vs No Promo")
    ax5.set_ylabel("Avg Sales")
    for i, v in enumerate(promo_avg.values):
        ax5.text(i, v + 2, f"{v:.1f}", ha='center', fontsize=10, color='white')

    # 6. Holiday vs Non-Holiday
    ax6 = fig.add_subplot(gs[2, 1])
    hol_avg = df.groupby('is_holiday')['sales'].mean()
    ax6.bar(['Regular Day', 'Holiday'], hol_avg.values,
            color=[PALETTE[0], PALETTE[3]], edgecolor='none', alpha=0.85)
    ax6.set_title("Sales: Holiday vs Regular Day")
    ax6.set_ylabel("Avg Sales")
    for i, v in enumerate(hol_avg.values):
        ax6.text(i, v + 2, f"{v:.1f}", ha='center', fontsize=10, color='white')

    # 7. Correlation heatmap
    ax7 = fig.add_subplot(gs[2, 2])
    num_cols = ['sales', 'price', 'is_weekend', 'is_holiday', 'is_promo',
                'month', 'day_of_week', 'quarter']
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax7, mask=mask, cmap='coolwarm', center=0,
                annot=True, fmt='.2f', annot_kws={'size': 7},
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax7.set_title("Feature Correlation Matrix")

    plt.savefig(f"{OUTPUT_DIR}/01_EDA.png", dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    plt.close()
    logger.info("EDA plot saved → 01_EDA.png")


run_eda(df_raw)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: TIME-SERIES DECOMPOSITION & STATIONARITY
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Performing time-series decomposition...")

def time_series_analysis(df: pd.DataFrame, product: str = "Product_A"):
    """Decompose a single product's time series and test stationarity."""
    ts = df[df['product'] == product].set_index('date')['sales']
    ts = ts.resample('W').sum()  # aggregate to weekly

    # ADF Test
    adf_result = adfuller(ts.dropna())
    print(f"\n[ADF Test — {product}]")
    print(f"  ADF Statistic : {adf_result[0]:.4f}")
    print(f"  p-value       : {adf_result[1]:.4f}")
    print(f"  Stationary    : {'Yes ✓' if adf_result[1] < 0.05 else 'No ✗'}")

    # Decompose
    decomp = seasonal_decompose(ts, model='additive', period=13)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle(f"Time-Series Decomposition — {product} (Weekly)",
                 fontsize=15, color='#e6edf3', fontweight='bold')

    for ax, data, label, color in zip(
            axes,
            [ts, decomp.trend, decomp.seasonal, decomp.resid],
            ['Original', 'Trend', 'Seasonality', 'Residuals'],
            [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]):
        ax.plot(data, color=color, lw=1.4)
        ax.fill_between(data.index, data, alpha=0.1, color=color)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        if label == 'Residuals':
            ax.axhline(0, color='white', lw=0.8, ls='--')

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_TimeSeries_Decomposition.png", dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    plt.close()
    logger.info("Time-series plot saved → 02_TimeSeries_Decomposition.png")

    return ts


ts_product_a = time_series_analysis(df_raw, "Product_A")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Engineering features for ML models...")

def engineer_features(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """Create lag features, rolling statistics, and calendar features."""
    d = df[df['product'] == product].copy()
    d.sort_values('date', inplace=True)
    d.reset_index(drop=True, inplace=True)

    # Lag features
    for lag in [1, 3, 7, 14, 21, 30]:
        d[f'lag_{lag}'] = d['sales'].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        d[f'roll_mean_{window}'] = d['sales'].shift(1).rolling(window).mean()
        d[f'roll_std_{window}'] = d['sales'].shift(1).rolling(window).std()
        d[f'roll_max_{window}'] = d['sales'].shift(1).rolling(window).max()
        d[f'roll_min_{window}'] = d['sales'].shift(1).rolling(window).min()

    # Calendar cyclical encoding
    d['sin_dow'] = np.sin(2 * np.pi * d['day_of_week'] / 7)
    d['cos_dow'] = np.cos(2 * np.pi * d['day_of_week'] / 7)
    d['sin_doy'] = np.sin(2 * np.pi * d['day_of_year'] / 365)
    d['cos_doy'] = np.cos(2 * np.pi * d['day_of_year'] / 365)
    d['sin_month'] = np.sin(2 * np.pi * d['month'] / 12)
    d['cos_month'] = np.cos(2 * np.pi * d['month'] / 12)

    # Day index for trend
    d['day_idx'] = (d['date'] - d['date'].min()).dt.days

    d.dropna(inplace=True)
    d.reset_index(drop=True, inplace=True)
    return d


FEATURE_COLS = [
    'lag_1', 'lag_3', 'lag_7', 'lag_14', 'lag_21', 'lag_30',
    'roll_mean_7', 'roll_std_7', 'roll_max_7', 'roll_min_7',
    'roll_mean_14', 'roll_std_14',
    'roll_mean_30', 'roll_std_30',
    'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy',
    'sin_month', 'cos_month',
    'is_weekend', 'is_holiday', 'is_promo',
    'quarter', 'day_idx', 'price'
]

TARGET = 'sales'

# Work with Product_A for detailed modelling
df_feat = engineer_features(df_raw, "Product_A")
X = df_feat[FEATURE_COLS].values
y = df_feat[TARGET].values

# Time-aware train/test split (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_train = df_feat['date'].values[:split]
dates_test  = df_feat['date'].values[split:]

# Scaling
scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s  = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_s  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: TRADITIONAL ML MODELS
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Training traditional ML models...")

results = {}

def evaluate_model(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape,
                     'y_pred': y_pred}
    print(f"  [{name:<28}] MAE={mae:8.2f} | RMSE={rmse:8.2f} | "
          f"R²={r2:.4f} | MAPE={mape:.2f}%")
    return y_pred


# 6.1 Linear Regression
print("\n── Linear / Regularized Regression ──")
for name, model in [
    ("Linear Regression",   LinearRegression()),
    ("Ridge Regression",    Ridge(alpha=1.0)),
    ("Lasso Regression",    Lasso(alpha=0.5, max_iter=5000)),
    ("ElasticNet",          ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000)),
]:
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    evaluate_model(name, y_test, pred)


# 6.2 KNN
print("\n── K-Nearest Neighbours (KNN) ──")
best_knn_score = np.inf
best_knn_k = 3
for k in [3, 5, 7, 10, 15]:
    knn = KNeighborsRegressor(n_neighbors=k, metric='minkowski', weights='distance')
    knn.fit(X_train_s, y_train)
    p = knn.predict(X_test_s)
    rmse = np.sqrt(mean_squared_error(y_test, p))
    if rmse < best_knn_score:
        best_knn_score = rmse
        best_knn_k = k

knn_best = KNeighborsRegressor(n_neighbors=best_knn_k, weights='distance')
knn_best.fit(X_train_s, y_train)
knn_pred = knn_best.predict(X_test_s)
evaluate_model(f"KNN (k={best_knn_k})", y_test, knn_pred)
print(f"  Best k selected: {best_knn_k}")


# 6.3 Decision Tree
print("\n── Decision Tree ──")
dt_model = DecisionTreeRegressor(
    max_depth=8, min_samples_split=10,
    min_samples_leaf=5, random_state=RANDOM_SEED
)
dt_model.fit(X_train_s, y_train)
dt_pred = dt_model.predict(X_test_s)
evaluate_model("Decision Tree", y_test, dt_pred)


# 6.4 Random Forest
print("\n── Random Forest ──")
rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=12, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt',
    random_state=RANDOM_SEED, n_jobs=-1
)
rf_model.fit(X_train_s, y_train)
rf_pred = rf_model.predict(X_test_s)
evaluate_model("Random Forest", y_test, rf_pred)


# 6.5 Gradient Boosting
print("\n── Gradient Boosting ──")
gb_model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=5,
    subsample=0.8, random_state=RANDOM_SEED
)
gb_model.fit(X_train_s, y_train)
gb_pred = gb_model.predict(X_test_s)
evaluate_model("Gradient Boosting", y_test, gb_pred)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7: SKLEARN ANN (MLPRegressor)
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Artificial Neural Network (sklearn MLP) ──")
ann_sklearn = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=RANDOM_SEED,
    verbose=False
)
ann_sklearn.fit(X_train_s, y_train)
ann_sk_pred = ann_sklearn.predict(X_test_s)
evaluate_model("ANN (sklearn MLP)", y_test, ann_sk_pred)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8: KERAS ANN (Deep Neural Network)
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Training Keras ANN...")
print("\n── ANN — Keras Deep Neural Network ──")

def build_ann(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,),
              kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse',
                  metrics=['mae'])
    return model


ann_keras = build_ann(X_train_s.shape[1])
ann_keras.summary()

callbacks_ann = [
    EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6, verbose=0)
]
history_ann = ann_keras.fit(
    X_train_s, y_train_s,
    validation_split=0.15,
    epochs=200,
    batch_size=32,
    callbacks=callbacks_ann,
    verbose=0
)
ann_pred_s = ann_keras.predict(X_test_s, verbose=0).ravel()
ann_pred   = scaler_y.inverse_transform(ann_pred_s.reshape(-1, 1)).ravel()
evaluate_model("ANN (Keras Deep)", y_test, ann_pred)
print(f"  Epochs trained: {len(history_ann.history['loss'])}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9: CNN FOR TIME-SERIES FORECASTING
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Training CNN for time-series forecasting...")
print("\n── CNN — 1D Convolutional Neural Network (Time-Series) ──")

LOOK_BACK = 30   # use 30 days of history

def create_sequences(X_arr, y_arr, look_back: int):
    Xs, ys = [], []
    for i in range(look_back, len(X_arr)):
        Xs.append(X_arr[i - look_back: i])
        ys.append(y_arr[i])
    return np.array(Xs), np.array(ys)


X_cnn, y_cnn = create_sequences(X_train_s, y_train_s, LOOK_BACK)
X_cnn_test, y_cnn_test = create_sequences(
    np.vstack([X_train_s[-LOOK_BACK:], X_test_s]),
    np.concatenate([y_train_s[-LOOK_BACK:], y_test_s]),
    LOOK_BACK
)

def build_cnn(time_steps: int, n_features: int) -> Sequential:
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
               input_shape=(time_steps, n_features)),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=5e-4), loss='mse', metrics=['mae'])
    return model


cnn_model = build_cnn(LOOK_BACK, X_cnn.shape[2])
cnn_model.summary()

callbacks_cnn = [
    EarlyStopping(patience=25, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(patience=12, factor=0.5, verbose=0)
]
history_cnn = cnn_model.fit(
    X_cnn, y_cnn,
    validation_split=0.15,
    epochs=150,
    batch_size=32,
    callbacks=callbacks_cnn,
    verbose=0
)
cnn_pred_s = cnn_model.predict(X_cnn_test, verbose=0).ravel()
cnn_pred   = scaler_y.inverse_transform(cnn_pred_s.reshape(-1, 1)).ravel()
y_cnn_true = scaler_y.inverse_transform(y_cnn_test.reshape(-1, 1)).ravel()
evaluate_model("CNN (1D Conv, Time-Series)", y_cnn_true, cnn_pred)
print(f"  Epochs trained: {len(history_cnn.history['loss'])}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10: HOLT-WINTERS EXPONENTIAL SMOOTHING (Time-Series)
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Holt-Winters Exponential Smoothing ──")
weekly_sales = df_raw[df_raw['product'] == 'Product_A'] \
                  .set_index('date')['sales'].resample('W').sum()

hw_train = weekly_sales.iloc[:int(len(weekly_sales) * 0.8)]
hw_test  = weekly_sales.iloc[int(len(weekly_sales) * 0.8):]

hw_model = ExponentialSmoothing(
    hw_train, seasonal='add', seasonal_periods=13,
    trend='add', damped_trend=True
).fit(optimized=True)

hw_pred = hw_model.forecast(len(hw_test))
hw_mae  = mean_absolute_error(hw_test.values, hw_pred.values)
hw_rmse = np.sqrt(mean_squared_error(hw_test.values, hw_pred.values))
print(f"  [Holt-Winters (Weekly)          ] MAE={hw_mae:8.2f} | RMSE={hw_rmse:8.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11: TRAINING HISTORY PLOTS (ANN & CNN)
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Plotting training histories...")

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle("Training History — ANN & CNN", fontsize=15, color='#e6edf3',
             fontweight='bold')

for ax, hist, label, color in zip(
        [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
        [history_ann, history_ann, history_cnn, history_cnn],
        ['ANN Loss (MSE)', 'ANN MAE', 'CNN Loss (MSE)', 'CNN MAE'],
        [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]):
    key = 'loss' if 'Loss' in label else 'mae'
    val_key = f'val_{key}'
    ax.plot(hist.history[key], color=color, lw=1.5, label='Train')
    ax.plot(hist.history[val_key], color='white', lw=1.5, ls='--', label='Val')
    ax.set_title(label)
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_Training_History.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
plt.close()
logger.info("Training history plot saved → 03_Training_History.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12: MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Generating model comparison plots...")

metrics_df = pd.DataFrame({
    name: {k: v for k, v in vals.items() if k != 'y_pred'}
    for name, vals in results.items()
}).T.reset_index().rename(columns={'index': 'Model'})

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle("Model Comparison Dashboard", fontsize=16, color='#e6edf3',
             fontweight='bold')

for ax, metric, color_idx in zip(
        axes.ravel(), ['MAE', 'RMSE', 'R2', 'MAPE'],
        [0, 1, 2, 3]):
    vals = metrics_df[metric].astype(float)
    bars = ax.barh(metrics_df['Model'], vals,
                   color=[PALETTE[i % len(PALETTE)] for i in range(len(metrics_df))],
                   edgecolor='none', height=0.6)
    ax.set_title(f"{metric} (lower is better)" if metric != 'R2'
                 else "R² Score (higher is better)")
    ax.set_xlabel(metric)
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', fontsize=8, color='#c9d1d9')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_Model_Comparison.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
plt.close()
logger.info("Model comparison saved → 04_Model_Comparison.png")

print("\n[RESULTS TABLE]")
print(metrics_df[['Model', 'MAE', 'RMSE', 'R2', 'MAPE']].to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 13: PREDICTION vs ACTUAL PLOTS
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Plotting predictions vs actuals...")

key_models = {
    "Random Forest":       results["Random Forest"]["y_pred"],
    "ANN (Keras Deep)":    results["ANN (Keras Deep)"]["y_pred"],
    "KNN (k=5)":           results.get(f"KNN (k={best_knn_k})", results["KNN (k=7)"] if "KNN (k=7)" in results else list(results.values())[5])["y_pred"],
    "Decision Tree":       results["Decision Tree"]["y_pred"],
}
# Find KNN key
knn_key = [k for k in results if k.startswith("KNN")][0]
key_models[knn_key] = results[knn_key]["y_pred"]

fig, axes = plt.subplots(2, 2, figsize=(22, 14))
fig.suptitle("Predicted vs Actual Sales — Key Models", fontsize=16,
             color='#e6edf3', fontweight='bold')

for ax, (name, pred) in zip(axes.ravel(), key_models.items()):
    ax.plot(dates_test, y_test, color='white', lw=1.2, label='Actual', alpha=0.8)
    ax.plot(dates_test, pred,   color=PALETTE[1], lw=1.2, label=name, alpha=0.85, ls='--')
    ax.fill_between(dates_test, y_test, pred, alpha=0.07, color=PALETTE[2])
    ax.set_title(name)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_Predictions_vs_Actual.png", dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
plt.close()
logger.info("Predictions plot saved → 05_Predictions_vs_Actual.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 14: DECISION TREE VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Visualizing Decision Tree...")

fig, ax = plt.subplots(figsize=(28, 12))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
plot_tree(
    dt_model,
    feature_names=FEATURE_COLS,
    max_depth=4,
    filled=True,
    rounded=True,
    fontsize=7,
    ax=ax,
    impurity=False,
    precision=1
)
ax.set_title("Decision Tree (max depth shown: 4)", fontsize=15,
             color='#e6edf3', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_Decision_Tree.png", dpi=120, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
plt.close()
logger.info("Decision tree plot saved → 06_Decision_Tree.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 15: FEATURE IMPORTANCE (Random Forest + Permutation)
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Computing feature importances...")

rf_importance = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS) \
                   .sort_values(ascending=False)

perm_imp = permutation_importance(rf_model, X_test_s, y_test,
                                   n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
perm_importance = pd.Series(perm_imp.importances_mean, index=FEATURE_COLS) \
                     .sort_values(ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle("Feature Importance Analysis", fontsize=15, color='#e6edf3',
             fontweight='bold')

top_n = 18
for ax, imp, title in [
    (ax1, rf_importance.head(top_n), "Random Forest Feature Importance"),
    (ax2, perm_importance.head(top_n), "Permutation Importance (RF)")
]:
    bars = ax.barh(imp.index[::-1], imp.values[::-1],
                   color=[PALETTE[i % len(PALETTE)] for i in range(top_n)],
                   edgecolor='none', height=0.65)
    ax.set_title(title)
    ax.set_xlabel("Importance Score")
    for bar, val in zip(bars, imp.values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', fontsize=8, color='#c9d1d9')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_Feature_Importance.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
plt.close()
logger.info("Feature importance saved → 07_Feature_Importance.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 16: KNN ANALYSIS — K vs RMSE
# ──────────────────────────────────────────────────────────────────────────────

logger.info("KNN — k selection analysis...")

k_vals, rmse_vals = [], []
for k in range(1, 31):
    m = KNeighborsRegressor(n_neighbors=k, weights='distance').fit(X_train_s, y_train)
    p = m.predict(X_test_s)
    k_vals.append(k)
    rmse_vals.append(np.sqrt(mean_squared_error(y_test, p)))

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(k_vals, rmse_vals, color=PALETTE[0], lw=2, marker='o', markersize=5)
ax.axvline(best_knn_k, color=PALETTE[2], ls='--', lw=1.5,
           label=f'Best k = {best_knn_k}')
ax.fill_between(k_vals, rmse_vals, alpha=0.1, color=PALETTE[0])
ax.set_title("KNN — k vs RMSE (Hyperparameter Tuning)", fontsize=14,
             color='#e6edf3', fontweight='bold')
ax.set_xlabel("k (Number of Neighbours)")
ax.set_ylabel("RMSE")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_KNN_k_vs_RMSE.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
plt.close()
logger.info("KNN analysis saved → 08_KNN_k_vs_RMSE.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 17: RANDOM FOREST — N_ESTIMATORS ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Random Forest — n_estimators analysis...")

n_trees, rf_rmse_vals, rf_r2_vals = [], [], []
for n in [10, 25, 50, 75, 100, 150, 200, 300]:
    m = RandomForestRegressor(n_estimators=n, max_depth=12,
                               random_state=RANDOM_SEED, n_jobs=-1).fit(X_train_s, y_train)
    p = m.predict(X_test_s)
    n_trees.append(n)
    rf_rmse_vals.append(np.sqrt(mean_squared_error(y_test, p)))
    rf_r2_vals.append(r2_score(y_test, p))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Random Forest — Hyperparameter Analysis", fontsize=14,
             color='#e6edf3', fontweight='bold')
ax1.plot(n_trees, rf_rmse_vals, color=PALETTE[1], lw=2, marker='s', markersize=7)
ax1.set_title("n_estimators vs RMSE")
ax1.set_xlabel("Number of Trees")
ax1.set_ylabel("RMSE")
ax1.grid(True, alpha=0.3)

ax2.plot(n_trees, rf_r2_vals, color=PALETTE[3], lw=2, marker='^', markersize=7)
ax2.set_title("n_estimators vs R²")
ax2.set_xlabel("Number of Trees")
ax2.set_ylabel("R² Score")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_RF_Hyperparameter.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
plt.close()
logger.info("RF hyperparameter plot saved → 09_RF_Hyperparameter.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 18: RESIDUAL ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Residual analysis...")

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle("Residual Analysis — Best Models", fontsize=15, color='#e6edf3',
             fontweight='bold')

model_preds = {
    "Random Forest": rf_pred,
    "ANN (Keras)":   ann_pred,
    "Decision Tree": dt_pred,
    "Gradient Boosting": gb_pred,
}
for ax, (name, pred) in zip(axes.ravel(), model_preds.items()):
    residuals = y_test - pred
    ax.scatter(pred, residuals, alpha=0.4, s=15,
               c=[PALETTE[i % len(PALETTE)] for i in range(len(pred))])
    ax.axhline(0, color='white', lw=1.2, ls='--')
    ax.set_title(f"Residuals — {name}")
    ax.set_xlabel("Predicted Sales")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.grid(True, alpha=0.3)

    # Annotate stats
    ax.text(0.05, 0.92, f"Mean={residuals.mean():.1f}\nStd={residuals.std():.1f}",
            transform=ax.transAxes, fontsize=9, color='white',
            bbox=dict(facecolor='#21262d', alpha=0.7, boxstyle='round'))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_Residual_Analysis.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
plt.close()
logger.info("Residual analysis saved → 10_Residual_Analysis.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 19: INVENTORY MANAGEMENT & SAFETY STOCK
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Computing inventory recommendations...")

def compute_inventory(df: pd.DataFrame, product: str,
                       lead_time_days: int = 7,
                       service_level_z: float = 1.645) -> dict:
    """
    Safety Stock = Z × σ_demand × sqrt(lead_time)
    Reorder Point = avg_demand × lead_time + safety_stock
    EOQ = sqrt(2 × D × S / H)
    """
    prod_df = df[df['product'] == product]
    avg_demand = prod_df['sales'].mean()
    std_demand = prod_df['sales'].std()
    annual_demand = avg_demand * 365

    safety_stock  = service_level_z * std_demand * np.sqrt(lead_time_days)
    reorder_point = avg_demand * lead_time_days + safety_stock

    # EOQ parameters
    ordering_cost = 500       # ₹ per order
    holding_cost  = 0.2 * avg_demand  # 20% of avg daily cost
    if holding_cost > 0:
        eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost)
    else:
        eoq = 0

    return {
        'product':         product,
        'avg_daily_sales': round(avg_demand, 2),
        'std_dev':         round(std_demand, 2),
        'lead_time_days':  lead_time_days,
        'service_level':   f"{service_level_z * 100 / 1.645 * 95:.0f}%",
        'safety_stock':    round(safety_stock, 2),
        'reorder_point':   round(reorder_point, 2),
        'EOQ':             round(eoq, 2),
        'annual_demand':   round(annual_demand, 2),
    }


inventory_results = []
for p in df_raw['product'].unique():
    inv = compute_inventory(df_raw, p, lead_time_days=7, service_level_z=1.645)
    inventory_results.append(inv)

inv_df = pd.DataFrame(inventory_results)
print("\n[INVENTORY RECOMMENDATIONS]")
print(inv_df.to_string(index=False))

# Inventory Dashboard
fig, axes = plt.subplots(2, 2, figsize=(20, 13))
fig.suptitle("Inventory Management Dashboard", fontsize=16, color='#e6edf3',
             fontweight='bold')

# Safety Stock
ax = axes[0, 0]
bars = ax.bar(inv_df['product'], inv_df['safety_stock'],
              color=PALETTE[:len(inv_df)], edgecolor='none', alpha=0.85)
ax.set_title("Safety Stock by Product")
ax.set_ylabel("Units")
for bar, val in zip(bars, inv_df['safety_stock']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.0f}", ha='center', fontsize=10, color='white')

# Reorder Point
ax = axes[0, 1]
x = np.arange(len(inv_df))
w = 0.35
ax.bar(x - w/2, inv_df['avg_daily_sales'] * 7, w, label='Lead-time Demand',
       color=PALETTE[0], alpha=0.8)
ax.bar(x + w/2, inv_df['reorder_point'], w, label='Reorder Point',
       color=PALETTE[2], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(inv_df['product'])
ax.set_title("Reorder Point vs Lead-Time Demand")
ax.set_ylabel("Units")
ax.legend()

# EOQ
ax = axes[1, 0]
bars = ax.bar(inv_df['product'], inv_df['EOQ'],
              color=PALETTE[3:3+len(inv_df)], edgecolor='none', alpha=0.85)
ax.set_title("Economic Order Quantity (EOQ)")
ax.set_ylabel("Units per Order")
for bar, val in zip(bars, inv_df['EOQ']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.0f}", ha='center', fontsize=10, color='white')

# Radar / spider chart for inventory metrics
ax = axes[1, 1]
categories = ['Avg Daily\nSales', 'Safety\nStock', 'Reorder\nPoint', 'EOQ/100']
for i, row in inv_df.iterrows():
    vals = [row['avg_daily_sales'], row['safety_stock'],
            row['reorder_point'], row['EOQ'] / 100]
    ax.plot(range(len(categories)), vals, marker='o',
            color=PALETTE[i], lw=1.5, label=row['product'])
    ax.fill_between(range(len(categories)), vals, alpha=0.08, color=PALETTE[i])
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=9)
ax.set_title("Multi-Metric Inventory Overview")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_Inventory_Dashboard.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
plt.close()
logger.info("Inventory dashboard saved → 11_Inventory_Dashboard.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 20: FUTURE DEMAND FORECASTING (30-day horizon)
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Generating 30-day future forecast...")

def forecast_future(model, last_known_X: np.ndarray, scaler_X, scaler_y,
                    n_days: int = 30) -> np.ndarray:
    """Iterative multi-step forecasting using last known features."""
    preds = []
    current_X = last_known_X.copy()
    for _ in range(n_days):
        x_s = scaler_X.transform(current_X.reshape(1, -1))
        p_s = model.predict(x_s)[0]
        preds.append(p_s)
        # Shift lag features
        for col_idx, col in enumerate(FEATURE_COLS):
            if col == 'lag_1':
                current_X[col_idx] = p_s
            elif col == 'lag_3':
                current_X[col_idx] = preds[-3] if len(preds) >= 3 else p_s
            elif col == 'lag_7':
                current_X[col_idx] = preds[-7] if len(preds) >= 7 else p_s
    return np.array(preds)


last_X = X_test[-1].copy()
future_preds_rf = forecast_future(rf_model, last_X, scaler_X, scaler_y, n_days=30)

last_date = df_feat['date'].max()
future_dates = [last_date + timedelta(days=i + 1) for i in range(30)]

# Confidence interval via bootstrap
bootstrap_preds = []
for _ in range(100):
    noise = np.random.normal(0, results["Random Forest"]["RMSE"] * 0.3,
                             size=len(future_preds_rf))
    bootstrap_preds.append(future_preds_rf + noise)
bootstrap_preds = np.array(bootstrap_preds)
ci_lower = np.percentile(bootstrap_preds, 5,  axis=0)
ci_upper = np.percentile(bootstrap_preds, 95, axis=0)

fig, ax = plt.subplots(figsize=(18, 8))
n_hist = 60
hist_dates = df_feat['date'].values[-n_hist:]
hist_sales  = df_feat['sales'].values[-n_hist:]

ax.plot(hist_dates, hist_sales, color='white', lw=1.5, label='Historical (last 60d)')
ax.axvline(last_date, color=PALETTE[3], ls='--', lw=1.5, label='Forecast start')
ax.plot(future_dates, future_preds_rf, color=PALETTE[1], lw=2,
        marker='o', markersize=4, label='30-Day Forecast (RF)')
ax.fill_between(future_dates, ci_lower, ci_upper, alpha=0.2,
                color=PALETTE[1], label='90% Confidence Interval')

ax.set_title("30-Day Sales Forecast — Random Forest", fontsize=14,
             color='#e6edf3', fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Sales")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_Future_Forecast.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
plt.close()
logger.info("Future forecast saved → 12_Future_Forecast.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 21: MULTI-PRODUCT FORECASTING COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Multi-product forecasting...")

fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
fig.suptitle("Multi-Product 30-Day Sales Forecast", fontsize=15, color='#e6edf3',
             fontweight='bold')

for ax, (product, color) in zip(axes, zip(df_raw['product'].unique(), PALETTE)):
    prod_ts = df_raw[df_raw['product'] == product].set_index('date')['sales']
    prod_ts_weekly = prod_ts.resample('W').sum()

    try:
        model_hw = ExponentialSmoothing(
            prod_ts_weekly, seasonal='add', seasonal_periods=13,
            trend='add', damped_trend=True
        ).fit(optimized=True)
        forecast = model_hw.forecast(4)  # 4 weeks ahead
        ax.plot(prod_ts_weekly.index[-20:], prod_ts_weekly.values[-20:],
                color='white', lw=1.2, label='Historical')
        ax.plot(forecast.index, forecast.values, color=color,
                lw=2, marker='o', markersize=5, label='Forecast')
        ax.fill_between(forecast.index,
                        forecast.values * 0.85,
                        forecast.values * 1.15,
                        alpha=0.2, color=color)
    except Exception:
        ax.text(0.5, 0.5, "Insufficient\nData", ha='center', va='center',
                transform=ax.transAxes)

    ax.set_title(product, fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales" if ax == axes[0] else "")
    ax.tick_params(axis='x', rotation=35)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_MultiProduct_Forecast.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
plt.close()
logger.info("Multi-product forecast saved → 13_MultiProduct_Forecast.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 22: CROSS-VALIDATION (Time-Series Split)
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Time-series cross-validation...")

tscv = TimeSeriesSplit(n_splits=5)
cv_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
    "Decision Tree":     DecisionTreeRegressor(max_depth=8, random_state=RANDOM_SEED),
    "KNN":               KNeighborsRegressor(n_neighbors=best_knn_k, weights='distance'),
}

cv_results = {}
for name, model in cv_models.items():
    scores = []
    for train_idx, val_idx in tscv.split(X_train_s):
        model.fit(X_train_s[train_idx], y_train[train_idx])
        p = model.predict(X_train_s[val_idx])
        scores.append(np.sqrt(mean_squared_error(y_train[val_idx], p)))
    cv_results[name] = scores
    print(f"  CV RMSE [{name:<22}]: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

fig, ax = plt.subplots(figsize=(14, 7))
ax.boxplot(cv_results.values(), patch_artist=True,
           boxprops={'facecolor': '#21262d'},
           medianprops={'color': 'white', 'lw': 2},
           whiskerprops={'color': '#8b949e'},
           capprops={'color': '#8b949e'})
for i, (patch, color) in enumerate(zip(ax.patches, PALETTE)):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(cv_results.keys(), rotation=20)
ax.set_title("5-Fold Time-Series Cross-Validation RMSE", fontsize=14,
             color='#e6edf3', fontweight='bold')
ax.set_ylabel("RMSE")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/14_CrossValidation.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
plt.close()
logger.info("Cross-validation plot saved → 14_CrossValidation.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 23: SUMMARY DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────

logger.info("Generating final summary dashboard...")

# Best model
best_model_name = metrics_df.loc[metrics_df['RMSE'].astype(float).idxmin(), 'Model']
best_rmse = float(metrics_df.loc[metrics_df['RMSE'].astype(float).idxmin(), 'RMSE'])
best_r2   = float(metrics_df.loc[metrics_df['RMSE'].astype(float).idxmin(), 'R2'])

fig = plt.figure(figsize=(24, 16))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Title
ax_title = fig.add_subplot(gs[0, :])
ax_title.set_facecolor('#161b22')
ax_title.text(0.5, 0.65, "BizForecast — Sales Demand Forecasting System",
              ha='center', va='center', fontsize=22, color='#58a6ff',
              fontweight='bold', transform=ax_title.transAxes)
ax_title.text(0.5, 0.25,
              f"Best Model: {best_model_name}  |  RMSE: {best_rmse:.2f}  |  R²: {best_r2:.4f}  |  "
              f"Products: {df_raw['product'].nunique()}  |  Days: {df_raw['date'].nunique()}",
              ha='center', va='center', fontsize=13, color='#8b949e',
              transform=ax_title.transAxes)
ax_title.axis('off')

# RMSE comparison (mini)
ax1 = fig.add_subplot(gs[1, :2])
rmse_vals_plot = metrics_df[['Model', 'RMSE']].copy()
rmse_vals_plot['RMSE'] = rmse_vals_plot['RMSE'].astype(float)
rmse_vals_plot.sort_values('RMSE', inplace=True)
bars = ax1.barh(rmse_vals_plot['Model'], rmse_vals_plot['RMSE'],
                color=[PALETTE[i % len(PALETTE)] for i in range(len(rmse_vals_plot))],
                edgecolor='none', height=0.6)
ax1.set_title("Model RMSE Comparison")
ax1.set_xlabel("RMSE")
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# R2 comparison
ax2 = fig.add_subplot(gs[1, 2:])
r2_vals_plot = metrics_df[['Model', 'R2']].copy()
r2_vals_plot['R2'] = r2_vals_plot['R2'].astype(float)
r2_vals_plot.sort_values('R2', ascending=False, inplace=True)
bars2 = ax2.barh(r2_vals_plot['Model'], r2_vals_plot['R2'],
                 color=[PALETTE[i % len(PALETTE)] for i in range(len(r2_vals_plot))],
                 edgecolor='none', height=0.6)
ax2.set_title("Model R² Score Comparison")
ax2.set_xlabel("R²")
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

# Safety stock summary
ax3 = fig.add_subplot(gs[2, :2])
x_pos = np.arange(len(inv_df))
ax3.bar(x_pos - 0.2, inv_df['safety_stock'], 0.35,
        label='Safety Stock', color=PALETTE[0], alpha=0.85)
ax3.bar(x_pos + 0.2, inv_df['reorder_point'], 0.35,
        label='Reorder Point', color=PALETTE[2], alpha=0.85)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(inv_df['product'])
ax3.set_title("Safety Stock & Reorder Points")
ax3.set_ylabel("Units")
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Top features
ax4 = fig.add_subplot(gs[2, 2:])
top_feat = rf_importance.head(12)
ax4.barh(top_feat.index[::-1], top_feat.values[::-1],
         color=[PALETTE[i % len(PALETTE)] for i in range(12)],
         edgecolor='none', height=0.65)
ax4.set_title("Top 12 Feature Importances (RF)")
ax4.set_xlabel("Importance")
ax4.grid(True, alpha=0.3, axis='x')

plt.savefig(f"{OUTPUT_DIR}/00_Summary_Dashboard.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
plt.close()
logger.info("Summary dashboard saved → 00_Summary_Dashboard.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 24: FINAL REPORT
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  BIZFORECAST — FINAL REPORT")
print("=" * 70)
print(f"\n  Dataset     : {df_raw.shape[0]} records | {df_raw['product'].nunique()} products | "
      f"{df_raw['date'].nunique()} days")
print(f"  Train/Test  : {len(X_train)} / {len(X_test)} samples")
print(f"  Features    : {len(FEATURE_COLS)} engineered features")
print(f"  CNN Look-back : {LOOK_BACK} days")
print(f"\n  MODELS TRAINED:")
print(f"    ✔ Linear Regression (+ Ridge, Lasso, ElasticNet)")
print(f"    ✔ KNN (best k = {best_knn_k})")
print(f"    ✔ Decision Tree")
print(f"    ✔ Random Forest (200 trees)")
print(f"    ✔ Gradient Boosting")
print(f"    ✔ ANN — sklearn MLP")
print(f"    ✔ ANN — Keras Deep Neural Network")
print(f"    ✔ CNN — 1D Convolutional (time-series)")
print(f"    ✔ Holt-Winters Exponential Smoothing")
print(f"\n  BEST MODEL   : {best_model_name}")
print(f"  Best RMSE    : {best_rmse:.2f}")
print(f"  Best R²      : {best_r2:.4f}")
print(f"\n  OUTPUT FILES :")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"    → {os.path.join(OUTPUT_DIR, f)}")
print("\n  INVENTORY RECOMMENDATIONS COMPUTED:")
print(inv_df[['product', 'avg_daily_sales', 'safety_stock',
              'reorder_point', 'EOQ']].to_string(index=False))
print("\n" + "=" * 70)
print("  Project: BizForecast | AI-ML ")
print("=" * 70)
