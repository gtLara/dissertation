import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="IBOV fractional tracking spread analysis."
)
parser.add_argument(
    "--n-sweep",
    type=int,
    default=20,
    metavar="N",
    help="Number of d values in the fractional sweep [0, 1] (default: 20).",
)
args = parser.parse_args()

if args.n_sweep < 2:
    parser.error(f"--n-sweep must be at least 2, got {args.n_sweep}")

N_SWEEP = args.n_sweep
print(f"\nFractional sweep: {N_SWEEP} values of d in [0, 1]")


# ── style and constants ───────────────────────────────────────────────────────

plt.style.use("ggplot")
plt.rcParams.update(
    {
        "font.weight":        "normal",
        "axes.titleweight":   "normal",
        "axes.labelweight":   "normal",
        "figure.titleweight": "normal",
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.dpi":         120,
    }
)

RAW_DATA_DIRECTORY      = Path("data/raw")
CONSTITUENTS_CACHE_PATH = RAW_DATA_DIRECTORY / "ibov_constituents.parquet"
START_DATE              = "2010-01-01"
MAX_MISSING_FRACTION    = 0.05
NUMBER_OF_ACF_LAGS      = 60
SIGNIFICANCE_LEVEL      = 0.05
ROLLING_TE_WINDOW       = 21
FRACDIFF_THRESHOLD      = 1e-5
MAX_FRACDIFF_LAGS       = 500

SWEEP_CMAP = plt.get_cmap("rainbow")

IBOV_CONSTITUENT_TICKERS = [
    "ABEV3.SA", "ALOS3.SA", "ASAI3.SA", "AZUL4.SA", "B3SA3.SA",
    "BBAS3.SA", "BBDC4.SA", "BBSE3.SA", "BPAC11.SA", "BRAP4.SA",
    "BRFS3.SA", "BRKM5.SA", "CMIG4.SA", "COGN3.SA", "CPFE3.SA",
    "CPLE6.SA", "CSAN3.SA", "CSNA3.SA", "CYRE3.SA", "EGIE3.SA",
    "ELET3.SA", "ELET6.SA", "EMBR3.SA", "EQTL3.SA", "FLRY3.SA",
    "GGBR4.SA", "GOAU4.SA", "HAPV3.SA", "HYPE3.SA", "ITSA4.SA",
    "ITUB4.SA", "JBSS3.SA", "KLBN11.SA", "LREN3.SA", "MGLU3.SA",
    "MRFG3.SA", "MRVE3.SA", "MULT3.SA", "NTCO3.SA", "PETR3.SA",
    "PETR4.SA", "PRIO3.SA", "RADL3.SA", "RAIL3.SA", "RDOR3.SA",
    "RENT3.SA", "SANB11.SA", "SLCE3.SA", "SUZB3.SA", "TAEE11.SA",
    "TIMS3.SA", "TOTS3.SA", "UGPA3.SA", "USIM5.SA", "VALE3.SA",
    "VBBR3.SA", "VIVT3.SA", "WEGE3.SA", "YDUQ3.SA",
]


# ── data helpers ──────────────────────────────────────────────────────────────

def download_or_load_constituents() -> pd.DataFrame:
    if CONSTITUENTS_CACHE_PATH.exists():
        print("Loading constituents from cache...")
        return pd.read_parquet(CONSTITUENTS_CACHE_PATH)
    print(
        f"Downloading {len(IBOV_CONSTITUENT_TICKERS)} constituents "
        f"from Yahoo Finance..."
    )
    raw = yf.download(
        IBOV_CONSTITUENT_TICKERS,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    close_prices = pd.DataFrame(
        {
            ticker: raw[ticker]["Close"]
            for ticker in IBOV_CONSTITUENT_TICKERS
            if ticker in raw.columns.get_level_values(0)
        }
    )
    close_prices.index = pd.to_datetime(close_prices.index)
    close_prices.index.name = "date"
    close_prices.to_parquet(CONSTITUENTS_CACHE_PATH)
    return close_prices


def drop_sparse_tickers(
    prices: pd.DataFrame,
    max_missing_fraction: float,
) -> pd.DataFrame:
    missing_fractions = prices.isna().mean()
    retained = missing_fractions[
        missing_fractions <= max_missing_fraction
    ].index
    return prices[retained]


def estimate_ols_weights_multivariate(
    dependent: np.ndarray,
    independent: np.ndarray,
) -> np.ndarray:
    design_matrix = np.column_stack(
        [np.ones(len(dependent)), independent]
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix, dependent, rcond=None
    )
    return coefficients


def run_augmented_dickey_fuller_test(
    series: pd.Series,
    label: str,
) -> dict:
    result = adfuller(series.dropna(), autolag="AIC")
    interpretation = (
        "stationary" if result[1] < SIGNIFICANCE_LEVEL else "non-stationary"
    )
    print(f"\n  {label}")
    print(f"    ADF statistic : {result[0]:.4f}")
    print(f"    p-value       : {result[1]:.4f}")
    print(
        f"    critical values: 1%={result[4]['1%']:.3f}  "
        f"5%={result[4]['5%']:.3f}  10%={result[4]['10%']:.3f}"
    )
    print(
        f"    conclusion    : {interpretation} "
        f"at {int(SIGNIFICANCE_LEVEL * 100)}% level"
    )
    return {"statistic": result[0], "p_value": result[1]}


# ── fractional filter ─────────────────────────────────────────────────────────

def _fracdiff_weights(
    d: float,
    threshold: float,
    max_lags: int = MAX_FRACDIFF_LAGS,
) -> np.ndarray:
    """
    Compute the (1-L)^d weight vector via the recursion
        w_0 = 1,  w_k = w_{k-1} * (k - 1 - d) / k
    truncated once |w_k| < threshold or k reaches max_lags.
    """
    weights = [1.0]
    k = 1
    while k < max_lags:
        w = weights[-1] * (k - 1 - d) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.array(weights)


def fractional_difference(
    series: pd.Series,
    d: float,
    threshold: float = FRACDIFF_THRESHOLD,
) -> pd.Series:
    """
    Apply the (1-L)^d filter to a series.

    d=0 returns the series unchanged.
    d=1 returns series.diff().dropna() exactly.
    The leading (n_lags - 1) observations are dropped.

    :param series: Input time series (e.g. log-prices).
    :type series: pandas.Series
    :param d: Differencing order in [0, 1].
    :type d: float
    :param threshold: Truncation threshold for the weight vector.
    :type threshold: float
    :returns: Fractionally differenced series with leading NaNs removed.
    :rtype: pandas.Series
    """
    weights = _fracdiff_weights(d, threshold)
    n_lags  = len(weights)
    values  = series.values
    n       = len(values)
    result  = np.full(n, np.nan)
    for t in range(n_lags - 1, n):
        result[t] = np.dot(
            weights,
            values[t: t - n_lags if t - n_lags >= 0 else None: -1],
        )
    return pd.Series(result, index=series.index).dropna()


def fractional_integrate(
    values: np.ndarray,
    d: float,
    threshold: float = FRACDIFF_THRESHOLD,
) -> np.ndarray:
    """
    Apply the inverse filter (1-L)^{-d} recursively.

    Solves (1-L)^d z = y for z via:
        z_t = y_t - sum_{k=1}^{K} w_k * z_{t-k}

    d=0 returns values unchanged.
    d=1 returns cumsum(values) exactly.

    :param values: Array in fractionally differenced space.
    :type values: numpy.ndarray
    :param d: Differencing order matching the forward filter.
    :type d: float
    :param threshold: Truncation threshold matching the forward filter.
    :type threshold: float
    :returns: Integrated array of the same length.
    :rtype: numpy.ndarray
    """
    weights = _fracdiff_weights(d, threshold)
    n       = len(values)
    result  = np.zeros(n)
    for t in range(n):
        result[t] = values[t]
        for k in range(1, min(t + 1, len(weights))):
            result[t] -= weights[k] * result[t - k]
    return result


# ── unified tracking spread estimator ────────────────────────────────────────

def fit_tracking_spread(
    index_log_price: pd.Series,
    constituent_log_prices: pd.DataFrame,
    d: float,
) -> dict:
    """
    Estimate the OLS tracking spread at fractional differencing order d.

    Applies (1-L)^d to the index and all constituents, aligns on common
    dates, fits OLS, then inverts with (1-L)^{-d} to bring residuals and
    fitted values back to log-price space.

    d=0 recovers the level-based regression exactly.
    d=1 recovers the return-based regression exactly.

    :param index_log_price: Log-price series of the benchmark index.
    :type index_log_price: pandas.Series
    :param constituent_log_prices: Log-price DataFrame, one column per ticker.
    :type constituent_log_prices: pandas.DataFrame
    :param d: Fractional differencing order in [0, 1].
    :type d: float
    :returns: Dictionary with keys: spread, residuals_filtered,
        fitted_log_price, r_squared, dates.
    :rtype: dict
    """
    fd_index = fractional_difference(index_log_price, d)
    fd_constituents = constituent_log_prices.apply(
        lambda col: fractional_difference(col, d)
    )

    common          = fd_index.index.intersection(fd_constituents.index)
    fd_index        = fd_index.loc[common]
    fd_constituents = fd_constituents.loc[common]

    coefficients = estimate_ols_weights_multivariate(
        fd_index.values,
        fd_constituents.values,
    )
    design_matrix = np.column_stack(
        [np.ones(len(fd_index)), fd_constituents.values]
    )

    fitted_filtered    = design_matrix @ coefficients
    residuals_filtered = fd_index.values - fitted_filtered
    r_squared = 1 - np.var(residuals_filtered) / np.var(fd_index.values)

    spread = pd.Series(
        fractional_integrate(residuals_filtered, d),
        index=common,
    )
    fitted_log_price = pd.Series(
        fractional_integrate(fitted_filtered, d),
        index=common,
    )

    return {
        "spread":             spread,
        "residuals_filtered": residuals_filtered,
        "fitted_log_price":   fitted_log_price,
        "r_squared":          r_squared,
        "dates":              common,
    }


# ── load and prepare data ─────────────────────────────────────────────────────

index_prices = pd.read_parquet(RAW_DATA_DIRECTORY / "index.parquet")
index_log_price = np.log(index_prices["close"])

raw_constituent_prices = download_or_load_constituents()
raw_constituent_prices = drop_sparse_tickers(
    raw_constituent_prices, MAX_MISSING_FRACTION
)
raw_constituent_prices = raw_constituent_prices.dropna()
raw_constituent_prices.index = pd.to_datetime(raw_constituent_prices.index)

common_dates = index_log_price.index.intersection(
    raw_constituent_prices.index
)
index_log_price        = index_log_price.loc[common_dates]
constituent_log_prices = np.log(raw_constituent_prices.loc[common_dates])

retained_tickers = constituent_log_prices.columns.tolist()
print(
    f"\nRetained {len(retained_tickers)} of "
    f"{len(IBOV_CONSTITUENT_TICKERS)} constituents"
)
print(
    f"Date range  : {common_dates.min().date()} "
    f"to {common_dates.max().date()}"
)
print(f"Observations: {len(common_dates)}")


# ── fit at d=0 and d=1 ────────────────────────────────────────────────────────

result_d0 = fit_tracking_spread(index_log_price, constituent_log_prices, d=0.0)
result_d1 = fit_tracking_spread(index_log_price, constituent_log_prices, d=1.0)

spread_level_based     = result_d0["spread"]
spread_return_based    = result_d1["spread"]
r_squared_levels       = result_d0["r_squared"]
r_squared_returns      = result_d1["r_squared"]
level_residuals        = result_d0["residuals_filtered"]
return_tracking_errors = result_d1["residuals_filtered"]

print("\n--- in-sample fit ---\n")
print(f"  return-based (d=1)  R-squared = {r_squared_returns:.6f}")
print(f"  level-based  (d=0)  R-squared = {r_squared_levels:.6f}")

print("\n--- spread summary statistics ---\n")
for label, spread in [
    ("return-based (d=1)", spread_return_based),
    ("level-based  (d=0)", spread_level_based),
]:
    print(
        f"  {label}  "
        f"mean={spread.mean():.4f}  std={spread.std():.4f}  "
        f"min={spread.min():.4f}  max={spread.max():.4f}"
    )

print("\n--- augmented Dickey-Fuller tests ---")
adf_return_based = run_augmented_dickey_fuller_test(
    spread_return_based, "return-based spread (d=1)"
)
adf_level_based = run_augmented_dickey_fuller_test(
    spread_level_based, "level-based spread (d=0)"
)

running_variance_return = spread_return_based.expanding(min_periods=2).var()
running_variance_level  = spread_level_based.expanding(min_periods=2).var()

acf_values_return = acf(
    spread_return_based.dropna(), nlags=NUMBER_OF_ACF_LAGS, fft=True
)
acf_values_level = acf(
    spread_level_based.dropna(), nlags=NUMBER_OF_ACF_LAGS, fft=True
)

confidence_band = 1.96 / np.sqrt(len(spread_return_based.dropna()))
color_cycle     = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ann_factor      = np.sqrt(252)


# ── Figure 1 : Price spreads ──────────────────────────────────────────────────

figure_1, axes_1 = plt.subplots(2, 1, figsize=(13, 8), sharey=True)
figure_1.suptitle(
    f"Price spread: return-based vs level-based tracking, "
    f"IBOV tracked with {len(retained_tickers)} constituents"
)

axes_1[0].plot(spread_return_based, color=color_cycle[0], linewidth=0.7)
axes_1[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
axes_1[0].set_ylabel("log-price spread")
axes_1[0].set_title(
    f"return-based (d=1)  ADF p={adf_return_based['p_value']:.3f}"
)

axes_1[1].plot(spread_level_based, color=color_cycle[1], linewidth=0.7)
axes_1[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
axes_1[1].set_ylabel("log-price spread")
axes_1[1].set_xlabel("date")
axes_1[1].set_title(
    f"level-based (d=0)  ADF p={adf_level_based['p_value']:.3f}"
)

figure_1.tight_layout()
plt.show()


# ── Figure 2 : Running variance ───────────────────────────────────────────────

rv_return_clean = running_variance_return.dropna()
rv_level_clean  = running_variance_level.dropna()
common_rv_dates = rv_return_clean.index.intersection(rv_level_clean.index)
rv_return_aligned = rv_return_clean.loc[common_rv_dates].values
rv_level_aligned  = rv_level_clean.loc[common_rv_dates].values
number_of_observations = np.arange(1, len(common_rv_dates) + 1)

positive_mask      = (rv_return_aligned > 0) & (rv_level_aligned > 0)
log_time           = np.log(number_of_observations[positive_mask])
log_variance_return = np.log(rv_return_aligned[positive_mask])
log_variance_level  = np.log(rv_level_aligned[positive_mask])

figure_2, axes_2 = plt.subplots(1, 2, figsize=(13, 5))
figure_2.suptitle("Running variance of the price spread")

axes_2[0].plot(
    rv_return_aligned, color=color_cycle[0], linewidth=0.9,
    label="return-based (d=1)",
)
axes_2[0].plot(
    rv_level_aligned, color=color_cycle[1], linewidth=0.9,
    label="level-based (d=0)",
)
axes_2[0].set_ylabel("variance")
axes_2[0].set_xlabel("trading days")
axes_2[0].set_title("linear scale")
axes_2[0].legend(framealpha=0.8)

reference_line = log_time - log_time[0] + log_variance_return[0]

axes_2[1].plot(
    log_time, log_variance_return, color=color_cycle[0], linewidth=0.9,
    label="return-based (d=1)",
)
axes_2[1].plot(
    log_time, log_variance_level, color=color_cycle[1], linewidth=0.9,
    label="level-based (d=0)",
)
axes_2[1].plot(
    log_time, reference_line, color="gray", linewidth=0.8, linestyle="--",
    label="slope 1 (random walk reference)",
)
axes_2[1].set_ylabel("log variance")
axes_2[1].set_xlabel("log trading days")
axes_2[1].set_title("log-log scale")
axes_2[1].legend(framealpha=0.8)

figure_2.tight_layout()
plt.show()


# ── Figure 3 : ACF ────────────────────────────────────────────────────────────

figure_3, axes_3 = plt.subplots(2, 1, figsize=(13, 7))
figure_3.suptitle("Sample autocorrelation function of the price spread")

lags = np.arange(NUMBER_OF_ACF_LAGS + 1)

axes_3[0].bar(
    lags, acf_values_return, color=color_cycle[0], alpha=0.75, width=0.8
)
axes_3[0].axhline(
    confidence_band, color="gray", linewidth=0.8, linestyle="--", label="95% CI"
)
axes_3[0].axhline(-confidence_band, color="gray", linewidth=0.8, linestyle="--")
axes_3[0].axhline(0, color="gray", linewidth=0.3)
axes_3[0].set_ylabel("autocorrelation")
axes_3[0].set_title(
    f"return-based spread (d=1)  ADF p={adf_return_based['p_value']:.3f}"
)
axes_3[0].legend(framealpha=0.8)

axes_3[1].bar(
    lags, acf_values_level, color=color_cycle[1], alpha=0.75, width=0.8
)
axes_3[1].axhline(
    confidence_band, color="gray", linewidth=0.8, linestyle="--", label="95% CI"
)
axes_3[1].axhline(-confidence_band, color="gray", linewidth=0.8, linestyle="--")
axes_3[1].axhline(0, color="gray", linewidth=0.3)
axes_3[1].set_ylabel("autocorrelation")
axes_3[1].set_xlabel("lag (trading days)")
axes_3[1].set_title(
    f"level-based spread (d=0)  ADF p={adf_level_based['p_value']:.3f}"
)
axes_3[1].legend(framealpha=0.8)

figure_3.tight_layout()
plt.show()


# ── Figure 4 : Portfolio performance vs index ─────────────────────────────────

def _normalise(s: pd.Series) -> pd.Series:
    return s - s.iloc[0]


dates_d0 = result_d0["dates"]
dates_d1 = result_d1["dates"]

index_norm_d0 = _normalise(index_log_price.loc[dates_d0])
index_norm_d1 = _normalise(index_log_price.loc[dates_d1])

fitted_levels_norm = _normalise(result_d0["fitted_log_price"])
cum_returns_norm   = _normalise(result_d1["fitted_log_price"])

te_return_std = return_tracking_errors.std()
te_level_std  = level_residuals.std()

te_return_series = pd.Series(return_tracking_errors, index=dates_d1)
te_level_series  = pd.Series(level_residuals,         index=dates_d0)

figure_4, axes_4 = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
figure_4.suptitle(
    f"Portfolio performance vs IBOV  "
    f"{len(retained_tickers)} constituents  "
    f"log-price normalised to 0 at inception"
)

axes_4[0].plot(
    index_norm_d1, color="black", linewidth=0.9, label="IBOV index", zorder=3
)
axes_4[0].plot(
    cum_returns_norm, color=color_cycle[0], linewidth=0.9, linestyle="--",
    label="return-based portfolio (d=1)", zorder=2,
)
axes_4[0].fill_between(
    dates_d1, index_norm_d1, cum_returns_norm,
    alpha=0.18, color=color_cycle[0], label="tracking gap",
)
axes_4[0].axhline(0, color="gray", linewidth=0.4, linestyle=":")
axes_4[0].set_ylabel("cumulative log-return")
axes_4[0].set_title(
    f"return-based (d=1)  "
    f"daily TE std = {te_return_std:.5f}  "
    f"ann. TE = {te_return_std * ann_factor:.4f}  "
    f"R^2 = {r_squared_returns:.6f}"
)
axes_4[0].legend(framealpha=0.8, loc="upper left")

axes_4[1].plot(
    index_norm_d0, color="black", linewidth=0.9, label="IBOV index", zorder=3
)
axes_4[1].plot(
    fitted_levels_norm, color=color_cycle[1], linewidth=0.9, linestyle="--",
    label="level-based portfolio (d=0)", zorder=2,
)
axes_4[1].fill_between(
    dates_d0, index_norm_d0, fitted_levels_norm,
    alpha=0.18, color=color_cycle[1], label="tracking gap",
)
axes_4[1].axhline(0, color="gray", linewidth=0.4, linestyle=":")
axes_4[1].set_ylabel("cumulative log-return")
axes_4[1].set_title(
    f"level-based (d=0)  "
    f"residual std = {te_level_std:.5f}  "
    f"ann. equiv. = {te_level_std * ann_factor:.4f}  "
    f"R^2 = {r_squared_levels:.6f}"
)
axes_4[1].legend(framealpha=0.8, loc="upper left")

axes_4[2].plot(
    te_return_series.rolling(ROLLING_TE_WINDOW).std() * ann_factor,
    color=color_cycle[0], linewidth=0.9,
    label=f"return-based  {ROLLING_TE_WINDOW}d rolling ann. TE",
)
axes_4[2].plot(
    te_level_series.rolling(ROLLING_TE_WINDOW).std() * ann_factor,
    color=color_cycle[1], linewidth=0.9,
    label=f"level-based   {ROLLING_TE_WINDOW}d rolling ann. TE",
)
axes_4[2].axhline(0, color="gray", linewidth=0.4, linestyle=":")
axes_4[2].set_ylabel("annualised tracking error")
axes_4[2].set_xlabel("date")
axes_4[2].set_title(
    f"{ROLLING_TE_WINDOW}-day rolling annualised tracking error"
)
axes_4[2].legend(framealpha=0.8, loc="upper left")

figure_4.tight_layout()
plt.show()


# ── Figure 5 : 2x2 fractional sweep ──────────────────────────────────────────

D_SWEEP_VALUES = np.linspace(0.0, 1.0, N_SWEEP)

print(f"\n--- fractional sweep  {N_SWEEP} values ---")

sweep_results = []
for d in D_SWEEP_VALUES:
    res     = fit_tracking_spread(index_log_price, constituent_log_prices, d)
    sp      = res["spread"]
    rv      = sp.expanding(min_periods=2).var()
    adf_out = adfuller(sp.dropna(), autolag="AIC")
    sweep_results.append({
        "d":      d,
        "spread": sp,
        "rv":     rv,
        "dates":  res["dates"],
        "adf_p":  float(adf_out[1]),
    })
    print(f"  d={d:.3f}  ADF p={adf_out[1]:.4f}")

norm_d     = plt.Normalize(vmin=0.0, vmax=1.0)
bar_width  = (D_SWEEP_VALUES[1] - D_SWEEP_VALUES[0]) * 0.8

with plt.style.context("dark_background"):
    figure_5, axes_5 = plt.subplots(
        2, 2, figsize=(15, 10),
        gridspec_kw={"hspace": 0.38, "wspace": 0.28},
    )
    figure_5.suptitle(
        f"Fractional differencing sweep  d in [0, 1]  "
        f"({N_SWEEP} values)  "
        f"{len(retained_tickers)} constituents",
        color="white",
    )

    ax_spread = axes_5[0, 0]
    ax_var    = axes_5[1, 0]
    ax_perf   = axes_5[0, 1]
    ax_adf    = axes_5[1, 1]

    ref_dates    = sweep_results[0]["dates"]
    idx_ref      = index_log_price.loc[ref_dates]
    idx_ref_norm = idx_ref - idx_ref.iloc[0]

    for res in sweep_results:
        clr = SWEEP_CMAP(norm_d(res["d"]))
        sp  = res["spread"]
        dt  = res["dates"]

        ax_spread.plot(sp,         color=clr, linewidth=0.6, alpha=0.9)
        ax_var.plot(res["rv"],     color=clr, linewidth=0.6, alpha=0.9)

        idx_aligned = index_log_price.loc[dt]
        port_norm   = (idx_aligned - sp) - (idx_aligned - sp).iloc[0]
        ax_perf.plot(port_norm, color=clr, linewidth=0.55, alpha=0.85)

    ax_perf.plot(
        idx_ref_norm, color="white", linewidth=1.1, alpha=0.9,
        label="IBOV index",
    )

    d_vals     = [r["d"] for r in sweep_results]
    adf_ps     = [r["adf_p"] for r in sweep_results]
    bar_colors = [SWEEP_CMAP(norm_d(d)) for d in d_vals]
    ax_adf.bar(d_vals, adf_ps, width=bar_width, color=bar_colors, alpha=0.9)
    ax_adf.axhline(
        SIGNIFICANCE_LEVEL, color="white", linewidth=0.9, linestyle="--",
        label=f"p = {SIGNIFICANCE_LEVEL}",
    )

    for ax in (ax_spread, ax_var, ax_perf):
        ax.axhline(0, color="white", linewidth=0.4, linestyle=":", alpha=0.4)

    ax_spread.set_ylabel("log-price spread")
    ax_spread.set_title("tracking spread")
    ax_var.set_ylabel("variance")
    ax_var.set_xlabel("date")
    ax_var.set_title("expanding variance (linear scale)")
    ax_perf.set_ylabel("cumulative log-return")
    ax_perf.set_title("portfolio vs index")
    ax_perf.legend(fontsize=8, framealpha=0.3)
    ax_adf.set_ylabel("ADF p-value")
    ax_adf.set_xlabel("d")
    ax_adf.set_title("stationarity of spread (ADF p-value)")
    ax_adf.legend(fontsize=8, framealpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=SWEEP_CMAP, norm=norm_d)
    sm.set_array([])
    figure_5.subplots_adjust(right=0.88)
    cbar_ax = figure_5.add_axes([0.91, 0.08, 0.02, 0.82])
    cbar    = figure_5.colorbar(sm, cax=cbar_ax)
    cbar.set_label("d", rotation=0, labelpad=10, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.show()