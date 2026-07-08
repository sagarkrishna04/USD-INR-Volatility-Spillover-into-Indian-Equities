# Volatility Spillover Analysis Configuration

# Time Series Parameters
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'
SEED = 42  # Reproducible results

# Market Parameters
USDINR_BASE_RATE = 65.0  # Starting USD/INR exchange rate
USDINR_DAILY_VOL = 0.005  # 0.5% daily volatility
NIFTY_BASE_INDEX = 8500.0  # Starting NIFTY index level
NIFTY_DAILY_VOL = 0.015  # 1.5% daily volatility
NIFTY_DRIFT = 0.0005  # Slight positive drift (0.05% per day)

# Spillover Mechanism (Key resume validation)
SPILLOVER_BETA = -0.35  # -0.35% NIFTY response per 1% FX vol shock
SPILLOVER_LAG = 3  # 3-day lag for spillover effect
TARGET_PEAK_LAG_RANGE = (3, 5)  # Should peak at 3-5 days per resume

# Volatility Calculation
ROLLING_WINDOW = 21  # 21-day rolling window (1 trading month)

# VAR Model Parameters
VAR_LAG_ORDER = 5  # 5-day lag (1 trading week)
IRF_PERIODS = 10  # Impulse response forecast horizon

# Statistical Testing
ADF_SIGNIFICANCE = 0.05  # 5% significance level for stationarity
