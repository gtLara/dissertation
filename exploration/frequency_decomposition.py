from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("ggplot")
plt.rcParams.update(
    {
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "figure.titleweight": "normal",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
    }
)

RAW_DATA_DIRECTORY = Path("data/raw")

SMOOTHING_SCALES = [5, 21, 63, 126, 252]

BAND_LABELS = [
    "sub-weekly",
    "weekly",
    "monthly",
    "quarterly",
    "semi-annual",
    "trend",
]


index_prices = pd.read_parquet(RAW_DATA_DIRECTORY / "index.parquet")
constituent_prices = pd.read_parquet(RAW_DATA_DIRECTORY / "constituent.parquet")

common_dates = index_prices.index.intersection(constituent_prices.index)
index_prices = index_prices.loc[common_dates]
constituent_prices = constituent_prices.loc[common_dates]

print(f"Common date range: {common_dates.min().date()} to {common_dates.max().date()}")
print(f"Number of common trading days: {len(common_dates)}")

index_log_price = np.log(index_prices["close"])
constituent_log_price = np.log(constituent_prices["close"])

index_log_return = index_log_price.diff().dropna()
constituent_log_return = constituent_log_price.diff().dropna()


def compute_moving_average_smoothings(
    log_price: pd.Series,
    smoothing_scales: list[int],
) -> dict[int, pd.Series]:
    """
    Compute a sequence of moving average smoothings of a log price series.

    :param log_price: Log price series indexed by date.
    :type log_price: pandas.Series
    :param smoothing_scales: Window sizes in trading days, in ascending order.
    :type smoothing_scales: list[int]
    :returns: Dictionary mapping each scale to its smoothed series.
    :rtype: dict[int, pandas.Series]
    """
    return {
        scale: log_price.rolling(window=scale, min_periods=scale).mean()
        for scale in smoothing_scales
    }


def compute_band_components(
    log_price: pd.Series,
    smoothings: dict[int, pd.Series],
    smoothing_scales: list[int],
    band_labels: list[str],
) -> dict[str, pd.Series]:
    """
    Decompose a log price series into additive frequency band components.

    Bands are formed by differencing adjacent moving average smoothings.
    The components sum exactly to the original log price series.

    :param log_price: The original log price series.
    :type log_price: pandas.Series
    :param smoothings: Moving average smoothings keyed by window size.
    :type smoothings: dict[int, pandas.Series]
    :param smoothing_scales: Window sizes in ascending order.
    :type smoothing_scales: list[int]
    :param band_labels: Labels for each band from highest to lowest frequency.
    :type band_labels: list[str]
    :returns: Dictionary mapping band labels to their component series.
    :rtype: dict[str, pandas.Series]
    """
    bands = {}
    bands[band_labels[0]] = log_price - smoothings[smoothing_scales[0]]
    for position in range(len(smoothing_scales) - 1):
        bands[band_labels[position + 1]] = (
            smoothings[smoothing_scales[position]]
            - smoothings[smoothing_scales[position + 1]]
        )
    bands[band_labels[-1]] = smoothings[smoothing_scales[-1]]
    return bands


def normalize_to_origin(series: pd.Series) -> pd.Series:
    """
    Subtract the first valid value so the series starts at zero.

    :param series: The input series.
    :type series: pandas.Series
    :returns: The series shifted to start at zero.
    :rtype: pandas.Series
    """
    return series - series.loc[series.first_valid_index()]


def normalize_to_unit_variance(series: pd.Series) -> pd.Series:
    """
    Scale a series to unit variance.

    :param series: The input series.
    :type series: pandas.Series
    :returns: The series divided by its standard deviation.
    :rtype: pandas.Series
    """
    return series / series.std()


index_smoothings = compute_moving_average_smoothings(
    index_log_price, SMOOTHING_SCALES
)
index_bands = compute_band_components(
    index_log_price, index_smoothings, SMOOTHING_SCALES, BAND_LABELS
)

constituent_smoothings = compute_moving_average_smoothings(
    constituent_log_price, SMOOTHING_SCALES
)
constituent_bands = compute_band_components(
    constituent_log_price, constituent_smoothings, SMOOTHING_SCALES, BAND_LABELS
)

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


figure_1, axes_1 = plt.subplots(
    len(BAND_LABELS), 1, figsize=(14, 12), sharex=True
)
figure_1.suptitle("Moving average band decomposition of log prices")

for axis, band_label in zip(axes_1, BAND_LABELS):
    index_component = index_bands[band_label].copy()
    constituent_component = constituent_bands[band_label].copy()

    if band_label == "trend":
        index_component = normalize_to_origin(index_component)
        constituent_component = normalize_to_origin(constituent_component)

    axis.plot(
        index_component,
        color=color_cycle[0],
        linewidth=0.7,
        label="IBOV",
    )
    axis.plot(
        constituent_component,
        color=color_cycle[1],
        linewidth=0.7,
        label="VALE3",
        alpha=0.85,
    )
    axis.set_ylabel(band_label, rotation=0, ha="right", labelpad=70)
    axis.axhline(0, color="gray", linewidth=0.4, linestyle="--")

axes_1[0].legend(loc="upper right", framealpha=0.8)
axes_1[-1].set_xlabel("date")
figure_1.tight_layout()
plt.show()


figure_2, axes_2 = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
figure_2.suptitle("Trend extraction and reconstitution, IBOV")

index_log_price_normalized = normalize_to_origin(index_log_price)
index_trend_normalized = normalize_to_origin(index_bands["trend"])

axes_2[0].plot(
    index_log_price_normalized,
    color="gray",
    linewidth=0.7,
    label="original log price",
)
axes_2[0].plot(
    index_trend_normalized,
    color=color_cycle[0],
    linewidth=1.2,
    label="trend (252-day moving average)",
)
axes_2[0].set_ylabel("log price")
axes_2[0].legend(loc="upper left", framealpha=0.8)

non_trend_band_labels = [label for label in BAND_LABELS if label != "trend"]
non_trend_sum = pd.DataFrame(
    {label: index_bands[label] for label in non_trend_band_labels}
).sum(axis=1)

axes_2[1].plot(
    index_log_price_normalized - index_trend_normalized,
    color="gray",
    linewidth=0.6,
    label="original minus trend",
)
axes_2[1].plot(
    non_trend_sum,
    color=color_cycle[1],
    linewidth=0.9,
    linestyle="--",
    label="sum of all non-trend bands",
)
axes_2[1].axhline(0, color="gray", linewidth=0.4, linestyle=":")
axes_2[1].set_ylabel("log price residual")
axes_2[1].set_xlabel("date")
axes_2[1].legend(loc="upper left", framealpha=0.8)

figure_2.tight_layout()
plt.show()





def compute_rolling_band_correlation(
    band_a: pd.Series,
    band_b: pd.Series,
    window_size: int,
) -> pd.Series:
    """
    Compute the rolling Pearson correlation between two band components.

    :param band_a: First band component series indexed by date.
    :type band_a: pandas.Series
    :param band_b: Second band component series indexed by date.
    :type band_b: pandas.Series
    :param window_size: Rolling window size in trading days.
    :type window_size: int
    :returns: Rolling correlation series with NaN where insufficient
        data exists.
    :rtype: pandas.Series
    """
    combined = pd.concat([band_a, band_b], axis=1).dropna()
    return combined.iloc[:, 0].rolling(window=window_size).corr(
        combined.iloc[:, 1]
    )


def compute_rolling_correlation_std(
    band_a: pd.Series,
    band_b: pd.Series,
    window_size: int,
) -> float:
    """
    Compute the standard deviation of a rolling band correlation series.

    A high value indicates the correlation estimate is unstable for the
    given window size, i.e., the window is too short relative to the
    period of the band.

    :param band_a: First band component series indexed by date.
    :type band_a: pandas.Series
    :param band_b: Second band component series indexed by date.
    :type band_b: pandas.Series
    :param window_size: Rolling window size in trading days.
    :type window_size: int
    :returns: Standard deviation of the rolling correlation series,
        computed over all valid (non-NaN) observations.
    :rtype: float
    """
    rolling_correlation = compute_rolling_band_correlation(
        band_a, band_b, window_size
    )
    return rolling_correlation.std()


BAND_PERIODS_IN_TRADING_DAYS = {
    "sub-weekly": 3,
    "weekly": 10,
    "monthly": 21,
    "quarterly": 63,
    "semi-annual": 126,
}

CYCLES_FOR_ADEQUATE_ESTIMATION = 5

adequate_window_sizes = {
    band_label: CYCLES_FOR_ADEQUATE_ESTIMATION * period
    for band_label, period in BAND_PERIODS_IN_TRADING_DAYS.items()
}

too_short_window_sizes = {
    band_label: period
    for band_label, period in BAND_PERIODS_IN_TRADING_DAYS.items()
}

print("\n--- uncertainty principle metrics ---\n")
print(
    f"{'band':<14}  {'period':>8}  {'short window':>14}  "
    f"{'adequate window':>16}  {'std (short)':>12}  "
    f"{'std (adequate)':>14}  {'std ratio':>10}  "
    f"{'noise floor (short)':>20}  {'noise floor (adequate)':>22}"
)
print("-" * 140)

metrics_records = []

for band_label, period in BAND_PERIODS_IN_TRADING_DAYS.items():
    short_window = too_short_window_sizes[band_label]
    adequate_window = adequate_window_sizes[band_label]

    rolling_short = compute_rolling_band_correlation(
        index_bands[band_label],
        constituent_bands[band_label],
        short_window,
    ).dropna()

    rolling_adequate = compute_rolling_band_correlation(
        index_bands[band_label],
        constituent_bands[band_label],
        adequate_window,
    ).dropna()

    std_short = rolling_short.std()
    std_adequate = rolling_adequate.std()
    noise_floor_short = 1.0 / (short_window ** 0.5)
    noise_floor_adequate = 1.0 / (adequate_window ** 0.5)

    metrics_records.append(
        {
            "band": band_label,
            "period": period,
            "short_window": short_window,
            "adequate_window": adequate_window,
            "std_short": std_short,
            "std_adequate": std_adequate,
            "std_ratio": std_short / std_adequate,
            "noise_floor_short": noise_floor_short,
            "noise_floor_adequate": noise_floor_adequate,
        }
    )

    print(
        f"{band_label:<14}  {period:>8}  {short_window:>14}  "
        f"{adequate_window:>16}  {std_short:>12.4f}  "
        f"{std_adequate:>14.4f}  {std_short / std_adequate:>10.2f}  "
        f"{noise_floor_short:>20.4f}  {noise_floor_adequate:>22.4f}"
    )

metrics = pd.DataFrame(metrics_records).set_index("band")

illustration_bands = ["sub-weekly", "monthly", "quarterly"]

figure_4, axes_4 = plt.subplots(
    len(illustration_bands),
    1,
    figsize=(14, 11),
    sharex=False,
)
figure_4.suptitle(
    "Rolling correlation between IBOV and VALE3: "
    "one-cycle window versus adequate window, with noise floor"
)

for axis, band_label in zip(axes_4, illustration_bands):
    short_window = too_short_window_sizes[band_label]
    adequate_window = adequate_window_sizes[band_label]
    noise_floor_short = metrics.loc[band_label, "noise_floor_short"]
    noise_floor_adequate = metrics.loc[band_label, "noise_floor_adequate"]

    rolling_short = compute_rolling_band_correlation(
        index_bands[band_label],
        constituent_bands[band_label],
        short_window,
    )
    rolling_adequate = compute_rolling_band_correlation(
        index_bands[band_label],
        constituent_bands[band_label],
        adequate_window,
    )

    axis.plot(
        rolling_short,
        color=color_cycle[0],
        linewidth=0.6,
        alpha=0.7,
        label=f"window = {short_window}d (one cycle)",
    )
    axis.fill_between(
        rolling_short.index,
        rolling_short - noise_floor_short,
        rolling_short + noise_floor_short,
        color=color_cycle[0],
        alpha=0.12,
    )

    axis.plot(
        rolling_adequate,
        color=color_cycle[1],
        linewidth=1.1,
        alpha=0.9,
        label=f"window = {adequate_window}d (five cycles)",
    )
    axis.fill_between(
        rolling_adequate.index,
        rolling_adequate - noise_floor_adequate,
        rolling_adequate + noise_floor_adequate,
        color=color_cycle[1],
        alpha=0.18,
    )

    axis.axhline(0, color="gray", linewidth=0.4, linestyle=":")
    axis.set_ylabel("rolling correlation")
    axis.set_title(
        f"{band_label} band   "
        f"period ~ {BAND_PERIODS_IN_TRADING_DAYS[band_label]}d   "
        f"noise floor: {noise_floor_short:.2f} vs {noise_floor_adequate:.2f}"
    )
    axis.set_ylim(-0.5, 1.3)
    axis.legend(loc="upper right", framealpha=0.8)

axes_4[-1].set_xlabel("date")
figure_4.tight_layout()
plt.show()


def compute_band_moments(
    series_a: pd.Series,
    series_b: pd.Series,
) -> dict[str, float]:
    """
    Compute variance, covariance, and correlation between two series
    over their common valid observations.

    :param series_a: First series indexed by date.
    :type series_a: pandas.Series
    :param series_b: Second series indexed by date.
    :type series_b: pandas.Series
    :returns: Dictionary with keys ``variance_a``, ``variance_b``,
        ``covariance``, and ``correlation``.
    :rtype: dict[str, float]
    """
    combined = pd.concat([series_a, series_b], axis=1).dropna()
    column_a = combined.iloc[:, 0]
    column_b = combined.iloc[:, 1]
    return {
        "variance_a": column_a.var(),
        "variance_b": column_b.var(),
        "covariance": column_a.cov(column_b),
        "correlation": column_a.corr(column_b),
    }


def compute_minimum_variance_weight(
    variance_a: float,
    variance_b: float,
    covariance_ab: float,
) -> float:
    """
    Compute the minimum variance portfolio weight on asset A, constrained
    to the probability simplex.

    Solves the unconstrained quadratic program and clips to [0, 1].
    The weight on asset B is the complement ``1 - w``.

    :param variance_a: Variance of asset A returns.
    :type variance_a: float
    :param variance_b: Variance of asset B returns.
    :type variance_b: float
    :param covariance_ab: Covariance between asset A and asset B returns.
    :type covariance_ab: float
    :returns: Optimal weight on asset A in [0, 1].
    :rtype: float
    """
    denominator = variance_a + variance_b - 2.0 * covariance_ab
    if denominator == 0.0:
        return 0.5
    unconstrained_weight = (variance_b - covariance_ab) / denominator
    return float(np.clip(unconstrained_weight, 0.0, 1.0))


def compute_portfolio_variance(
    weight_a: float,
    variance_a: float,
    variance_b: float,
    covariance_ab: float,
) -> float:
    """
    Compute the variance of a two-asset portfolio.

    :param weight_a: Weight on asset A.
    :type weight_a: float
    :param variance_a: Variance of asset A.
    :type variance_a: float
    :param variance_b: Variance of asset B.
    :type variance_b: float
    :param covariance_ab: Covariance between assets A and B.
    :type covariance_ab: float
    :returns: Portfolio variance.
    :rtype: float
    """
    weight_b = 1.0 - weight_a
    return (
        weight_a ** 2 * variance_a
        + weight_b ** 2 * variance_b
        + 2.0 * weight_a * weight_b * covariance_ab
    )


analysis_band_labels = [
    label for label in BAND_LABELS if label != "trend"
]

VISUALIZATION_WINDOW_SIZE = 252


def compute_rolling_minimum_variance_weight(
    series_a: pd.Series,
    series_b: pd.Series,
    window_size: int,
) -> pd.Series:
    """
    Compute the time-varying minimum variance portfolio weight on asset A,
    using a rolling window to estimate second moments locally.

    At each point in time the weight is the closed-form solution to the
    two-asset minimum variance program constrained to the probability
    simplex, with all moments estimated from a rolling window of length
    ``window_size``.

    :param series_a: Band component or return series for asset A.
    :type series_a: pandas.Series
    :param series_b: Band component or return series for asset B.
    :type series_b: pandas.Series
    :param window_size: Rolling estimation window in trading days.
    :type window_size: int
    :returns: Time series of optimal weights on asset A, clipped to [0, 1].
    :rtype: pandas.Series
    """
    combined = pd.concat([series_a, series_b], axis=1).dropna()
    column_a = combined.iloc[:, 0]
    column_b = combined.iloc[:, 1]
    rolling_variance_a = column_a.rolling(window=window_size).var()
    rolling_variance_b = column_b.rolling(window=window_size).var()
    rolling_covariance = column_a.rolling(window=window_size).cov(column_b)
    denominator = (
        rolling_variance_a + rolling_variance_b - 2.0 * rolling_covariance
    )
    unconstrained_weight = (
        rolling_variance_b - rolling_covariance
    ) / denominator
    return unconstrained_weight.clip(0.0, 1.0)


band_rolling_correlations = {
    band_label: compute_rolling_band_correlation(
        index_bands[band_label],
        constituent_bands[band_label],
        VISUALIZATION_WINDOW_SIZE,
    )
    for band_label in analysis_band_labels
}

band_rolling_weights = {
    band_label: compute_rolling_minimum_variance_weight(
        index_bands[band_label],
        constituent_bands[band_label],
        VISUALIZATION_WINDOW_SIZE,
    )
    for band_label in analysis_band_labels
}

full_spectrum_rolling_weight = compute_rolling_minimum_variance_weight(
    index_log_return,
    constituent_log_return,
    VISUALIZATION_WINDOW_SIZE,
)

full_spectrum_rolling_correlation = compute_rolling_band_correlation(
    index_log_return,
    constituent_log_return,
    VISUALIZATION_WINDOW_SIZE,
)

print("\n--- rolling minimum variance portfolio, 252-day window ---\n")
print(
    f"{'band':<14}  {'mean corr':>10}  {'std corr':>10}  "
    f"{'mean w(IBOV)':>14}  {'std w(IBOV)':>12}  "
    f"{'min w':>8}  {'max w':>8}"
)
print("-" * 88)

for band_label in analysis_band_labels:
    rolling_correlation = band_rolling_correlations[band_label].dropna()
    rolling_weight = band_rolling_weights[band_label].dropna()
    print(
        f"{band_label:<14}  {rolling_correlation.mean():>10.4f}  "
        f"{rolling_correlation.std():>10.4f}  "
        f"{rolling_weight.mean():>14.4f}  "
        f"{rolling_weight.std():>12.4f}  "
        f"{rolling_weight.min():>8.4f}  "
        f"{rolling_weight.max():>8.4f}"
    )

full_spectrum_correlation_clean = full_spectrum_rolling_correlation.dropna()
full_spectrum_weight_clean = full_spectrum_rolling_weight.dropna()
print(
    f"{'full spectrum':<14}  "
    f"{full_spectrum_correlation_clean.mean():>10.4f}  "
    f"{full_spectrum_correlation_clean.std():>10.4f}  "
    f"{full_spectrum_weight_clean.mean():>14.4f}  "
    f"{full_spectrum_weight_clean.std():>12.4f}  "
    f"{full_spectrum_weight_clean.min():>8.4f}  "
    f"{full_spectrum_weight_clean.max():>8.4f}"
)

figure_weights, (axis_correlation, axis_weight) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True
)
figure_weights.suptitle(
    "Time-varying band correlations and minimum variance weights, "
    "IBOV and VALE3, 252-day rolling window"
)

for band_label, color in zip(analysis_band_labels, color_cycle):
    axis_correlation.plot(
        band_rolling_correlations[band_label],
        color=color,
        linewidth=0.8,
        alpha=0.85,
        label=band_label,
    )

axis_correlation.plot(
    full_spectrum_rolling_correlation,
    color="black",
    linewidth=1.0,
    linestyle="--",
    alpha=0.7,
    label="full spectrum",
)
axis_correlation.axhline(0, color="gray", linewidth=0.4, linestyle=":")
axis_correlation.set_ylabel("rolling correlation")
axis_correlation.set_title("band correlation between IBOV and VALE3")
axis_correlation.legend(loc="lower left", framealpha=0.8, ncol=3)

for band_label, color in zip(analysis_band_labels, color_cycle):
    axis_weight.plot(
        band_rolling_weights[band_label],
        color=color,
        linewidth=0.8,
        alpha=0.85,
        label=band_label,
    )

axis_weight.plot(
    full_spectrum_rolling_weight,
    color="black",
    linewidth=1.2,
    linestyle="--",
    alpha=0.9,
    label="full spectrum",
)
axis_weight.axhline(0.5, color="gray", linewidth=0.4, linestyle=":")
axis_weight.set_ylabel("weight on IBOV")
axis_weight.set_xlabel("date")
axis_weight.set_title(
    "minimum variance weight on IBOV by frequency band"
)
axis_weight.legend(loc="lower left", framealpha=0.8, ncol=3)
axis_weight.set_ylim(-0.05, 1.05)

figure_weights.tight_layout()
plt.show()


ONE_WAY_TRANSACTION_COST = 0.0010

REBALANCING_FREQUENCIES_IN_DAYS = {
    "daily": 1,
    "weekly": 5,
    "monthly": 21,
    "quarterly": 63,
}

TRADING_DAYS_PER_YEAR = 252

index_simple_return = np.exp(index_log_return) - 1
constituent_simple_return = np.exp(constituent_log_return) - 1


def simulate_portfolio_with_rebalancing(
    returns_a: pd.Series,
    returns_b: pd.Series,
    target_weights_a: pd.Series,
    rebalancing_frequency: int,
    one_way_cost: float,
) -> pd.DataFrame:
    """
    Simulate a two-asset portfolio with periodic rebalancing and
    proportional transaction costs.

    At each rebalancing date the portfolio is traded from its
    price-drifted weight to the current target. Between rebalancing
    dates the weight drifts passively with asset prices. Target weights
    are lagged by one trading day to prevent look-ahead bias.

    :param returns_a: Daily simple returns for asset A.
    :type returns_a: pandas.Series
    :param returns_b: Daily simple returns for asset B.
    :type returns_b: pandas.Series
    :param target_weights_a: Rolling optimal weight for asset A.
    :type target_weights_a: pandas.Series
    :param rebalancing_frequency: Trading days between rebalancing.
    :type rebalancing_frequency: int
    :param one_way_cost: Proportional cost per unit of turnover.
    :type one_way_cost: float
    :returns: DataFrame with ``portfolio_return``, ``held_weight``,
        and ``turnover_cost`` columns.
    :rtype: pandas.DataFrame
    """
    lagged_weights = target_weights_a.shift(1)
    combined = pd.concat(
        [returns_a, returns_b, lagged_weights], axis=1
    ).dropna()
    combined.columns = ["return_a", "return_b", "target_weight"]

    n = len(combined)
    portfolio_returns = np.empty(n)
    held_weights = np.empty(n)
    turnover_costs = np.zeros(n)

    held_weight = float(combined["target_weight"].iloc[0])

    for i in range(n):
        r_a = combined["return_a"].iloc[i]
        r_b = combined["return_b"].iloc[i]
        target = combined["target_weight"].iloc[i]

        gross_return = held_weight * r_a + (1.0 - held_weight) * r_b
        denominator = 1.0 + gross_return
        drifted_weight = (
            held_weight * (1.0 + r_a) / denominator
            if denominator != 0.0
            else held_weight
        )

        cost = 0.0
        if i > 0 and i % rebalancing_frequency == 0 and not np.isnan(target):
            turnover = abs(target - drifted_weight)
            cost = turnover * one_way_cost
            held_weight = target
        else:
            held_weight = drifted_weight

        portfolio_returns[i] = gross_return - cost
        held_weights[i] = held_weight
        turnover_costs[i] = cost

    return pd.DataFrame(
        {
            "portfolio_return": portfolio_returns,
            "held_weight": held_weights,
            "turnover_cost": turnover_costs,
        },
        index=combined.index,
    )


def compute_performance_metrics(
    simulation_result: pd.DataFrame,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """
    Compute annualised performance metrics from a portfolio simulation.

    :param simulation_result: Output of
        :func:`simulate_portfolio_with_rebalancing`.
    :type simulation_result: pandas.DataFrame
    :param trading_days_per_year: Trading days used to annualise.
    :type trading_days_per_year: int
    :returns: Dictionary with annualised return, volatility, Sharpe
        ratio, annual turnover, and total cost in basis points.
    :rtype: dict[str, float]
    """
    returns = simulation_result["portfolio_return"].dropna()
    costs = simulation_result["turnover_cost"]
    years = len(returns) / trading_days_per_year

    annualized_return = returns.mean() * trading_days_per_year
    annualized_volatility = returns.std() * np.sqrt(trading_days_per_year)
    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility > 0.0
        else np.nan
    )

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "annual_turnover": costs.sum() / years,
        "total_cost_bps": costs.sum() * 10_000.0,
    }


all_weight_series = {
    **band_rolling_weights,
    "full spectrum": full_spectrum_rolling_weight,
}

portfolio_labels_ordered = list(analysis_band_labels) + ["full spectrum"]

simulation_results = {
    portfolio_label: {
        rebalancing_label: simulate_portfolio_with_rebalancing(
            returns_a=index_simple_return,
            returns_b=constituent_simple_return,
            target_weights_a=weight_series,
            rebalancing_frequency=rebalancing_frequency,
            one_way_cost=ONE_WAY_TRANSACTION_COST,
        )
        for rebalancing_label, rebalancing_frequency
        in REBALANCING_FREQUENCIES_IN_DAYS.items()
    }
    for portfolio_label, weight_series in all_weight_series.items()
}

performance_metrics = {
    portfolio_label: {
        rebalancing_label: compute_performance_metrics(simulation)
        for rebalancing_label, simulation in rebalancing_results.items()
    }
    for portfolio_label, rebalancing_results in simulation_results.items()
}

cost_label = f"{int(ONE_WAY_TRANSACTION_COST * 10_000)}bp one-way cost"

for rebalancing_label in REBALANCING_FREQUENCIES_IN_DAYS:
    print(f"\n--- {rebalancing_label} rebalancing, {cost_label} ---\n")
    print(
        f"{'portfolio':<14}  {'ann return':>11}  {'ann vol':>9}  "
        f"{'sharpe':>8}  {'ann turnover':>13}  {'total cost bps':>15}"
    )
    print("-" * 82)
    for label in portfolio_labels_ordered:
        metrics = performance_metrics[label][rebalancing_label]
        print(
            f"{label:<14}  "
            f"{metrics['annualized_return']:>10.2%}  "
            f"{metrics['annualized_volatility']:>9.2%}  "
            f"{metrics['sharpe_ratio']:>8.3f}  "
            f"{metrics['annual_turnover']:>13.4f}  "
            f"{metrics['total_cost_bps']:>15.1f}"
        )

portfolio_colors = {
    label: color_cycle[i] for i, label in enumerate(analysis_band_labels)
}

ibov_cumulative_wealth = (1.0 + index_simple_return).cumprod()
vale3_cumulative_wealth = (1.0 + constituent_simple_return).cumprod()

figure_performance, axes_performance = plt.subplots(
    2, 2, figsize=(14, 10), sharex=False
)
figure_performance.suptitle(
    f"Out-of-sample cumulative wealth by band and rebalancing frequency "
    f"({cost_label})"
)

for axis, rebalancing_label in zip(
    axes_performance.flatten(),
    REBALANCING_FREQUENCIES_IN_DAYS.keys(),
):
    for band_label in analysis_band_labels:
        simulation = simulation_results[band_label][rebalancing_label]
        cumulative_wealth = (1.0 + simulation["portfolio_return"]).cumprod()
        axis.plot(
            cumulative_wealth,
            color=portfolio_colors[band_label],
            linewidth=0.9,
            alpha=0.85,
            label=band_label,
        )

    full_spectrum_simulation = simulation_results["full spectrum"][rebalancing_label]
    full_spectrum_wealth = (
        1.0 + full_spectrum_simulation["portfolio_return"]
    ).cumprod()
    axis.plot(
        full_spectrum_wealth,
        color="black",
        linewidth=1.2,
        linestyle="--",
        alpha=0.9,
        label="full spectrum",
    )

    axis.plot(
        ibov_cumulative_wealth,
        color="gray",
        linewidth=0.6,
        linestyle=":",
        alpha=0.5,
        label="IBOV",
    )
    axis.plot(
        vale3_cumulative_wealth,
        color="gray",
        linewidth=0.6,
        linestyle="-.",
        alpha=0.5,
        label="VALE3",
    )

    axis.set_title(f"{rebalancing_label} rebalancing")
    axis.set_ylabel("cumulative wealth")
    axis.set_xlabel("date")
    axis.set_yscale("log")

axes_performance.flatten()[0].legend(
    loc="upper left", framealpha=0.8, fontsize=8, ncol=2
)

figure_performance.tight_layout()
plt.show()