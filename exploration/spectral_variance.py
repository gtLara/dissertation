from pathlib import Path
from typing import Callable

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy import stats


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
ROLLING_WINDOW_SIZE = 750
NUMBER_OF_FREQUENCY_BANDS = 15
TRADING_DAYS_PER_YEAR = 252
WELCH_SUB_SEGMENT_LENGTH = 375
WELCH_OVERLAP_FRACTION = 0.5
DWT_WAVELET_NAME = "db4"
MORLET_CENTRAL_FREQUENCY = 6.0
MORLET_SCALES_PER_OCTAVE = 10
WAVELET_DIAGNOSTIC_N_LAGS = 20
WAVELET_TFR_SMOOTHING_WINDOW = 21
HIGH_MID_FREQUENCY_BOUNDARY_IN_TRADING_DAYS = 5.0
MID_LOW_FREQUENCY_BOUNDARY_IN_TRADING_DAYS = 21.0
NUMBER_OF_WHITE_NOISE_PERMUTATIONS = 2000

CRISIS_DATES_AND_LABELS = {
    "1999-01-15": "BRL float",
    "2002-09-01": "Lula election / Argentina default",
    "2008-09-15": "Lehman collapse",
    "2020-03-11": "COVID-19",
}

IMPORTANT_HORIZON_TICKS_IN_TRADING_DAYS = {
    2: "2d",
    5: "1w",
    21: "1m",
    63: "1q",
    126: "6m",
    252: "1y",
    504: "2y",
    750: "3y",
}


def load_index_log_returns(raw_data_directory: Path) -> pd.Series:
    index_prices = pd.read_parquet(raw_data_directory / "index.parquet")
    index_log_price = np.log(index_prices["close"])
    index_log_return = index_log_price.diff().dropna()
    return index_log_return


def compute_annualized_variance(window_returns: pd.Series) -> float:
    return float(window_returns.var(ddof=0) * TRADING_DAYS_PER_YEAR)


def compute_one_sided_annualized_variance_spectrum_rectangular(
    window_returns: pd.Series,
) -> np.ndarray:
    window_size = len(window_returns)
    demeaned_returns = window_returns.values - window_returns.values.mean()
    dft_coefficients = np.fft.fft(demeaned_returns)
    squared_magnitudes = np.abs(dft_coefficients) ** 2 / window_size ** 2
    one_sided_spectrum = squared_magnitudes[1 : window_size // 2 + 1].copy()
    one_sided_spectrum[: window_size // 2 - 1] *= 2
    return one_sided_spectrum * TRADING_DAYS_PER_YEAR


def compute_one_sided_annualized_variance_spectrum_welch(
    window_returns: pd.Series,
) -> np.ndarray:
    demeaned_returns = window_returns.values - window_returns.values.mean()
    sub_segment_length = WELCH_SUB_SEGMENT_LENGTH
    step_size = int(sub_segment_length * (1.0 - WELCH_OVERLAP_FRACTION))
    hann_window = np.hanning(sub_segment_length)
    window_power = np.sum(hann_window ** 2)
    segment_start_indices = range(
        0, len(demeaned_returns) - sub_segment_length + 1, step_size
    )
    sub_segment_periodograms = []
    for start_index in segment_start_indices:
        sub_segment = demeaned_returns[start_index : start_index + sub_segment_length]
        tapered_sub_segment = hann_window * sub_segment
        dft_coefficients = np.fft.fft(tapered_sub_segment)
        squared_magnitudes = (
            np.abs(dft_coefficients) ** 2 / (window_power * sub_segment_length)
        )
        one_sided = squared_magnitudes[1 : sub_segment_length // 2 + 1].copy()
        one_sided[: sub_segment_length // 2 - 1] *= 2
        sub_segment_periodograms.append(one_sided)
    averaged_spectrum = np.mean(sub_segment_periodograms, axis=0)
    return averaged_spectrum * TRADING_DAYS_PER_YEAR


def compute_log_bandwidth_overlap_fraction(
    source_lower_period: float,
    source_upper_period: float,
    target_lower_period: float,
    target_upper_period: float,
) -> float:
    overlap_lower = max(source_lower_period, target_lower_period)
    overlap_upper = min(source_upper_period, target_upper_period)
    if overlap_upper <= overlap_lower:
        return 0.0
    overlap_log_width = np.log(overlap_upper) - np.log(overlap_lower)
    source_log_width = np.log(source_upper_period) - np.log(source_lower_period)
    return overlap_log_width / source_log_width


def compute_dwt_band_variances_for_window(
    window_returns: pd.Series,
    period_band_edges: np.ndarray,
) -> np.ndarray:
    window_size = len(window_returns)
    demeaned = window_returns.values - window_returns.values.mean()
    max_level = pywt.dwt_max_level(window_size, DWT_WAVELET_NAME)
    coefficients = pywt.wavedec(
        demeaned, DWT_WAVELET_NAME, level=max_level, mode="periodization"
    )
    number_of_bands = len(period_band_edges) - 1
    band_variances = np.zeros(number_of_bands)

    for level in range(1, max_level + 1):
        coeff_index = max_level - level + 1
        detail_coefficients = coefficients[coeff_index]
        level_variance = (
            np.sum(detail_coefficients ** 2) / window_size * TRADING_DAYS_PER_YEAR
        )
        level_lower_period = 2.0 ** level
        level_upper_period = 2.0 ** (level + 1)
        for band_index in range(number_of_bands):
            overlap = compute_log_bandwidth_overlap_fraction(
                level_lower_period,
                level_upper_period,
                period_band_edges[band_index],
                period_band_edges[band_index + 1],
            )
            band_variances[band_index] += level_variance * overlap

    approximation_coefficients = coefficients[0]
    approximation_variance = (
        np.sum(approximation_coefficients ** 2) / window_size * TRADING_DAYS_PER_YEAR
    )
    approximation_lower_period = 2.0 ** (max_level + 1)
    approximation_upper_period = period_band_edges[-1]

    if approximation_lower_period < approximation_upper_period:
        for band_index in range(number_of_bands):
            overlap = compute_log_bandwidth_overlap_fraction(
                approximation_lower_period,
                approximation_upper_period,
                period_band_edges[band_index],
                period_band_edges[band_index + 1],
            )
            band_variances[band_index] += approximation_variance * overlap
    else:
        band_variances[-1] += approximation_variance

    return band_variances


def build_log_spaced_period_band_edges(window_size: int, number_of_bands: int) -> np.ndarray:
    return np.logspace(
        np.log10(2.0),
        np.log10(float(window_size)),
        number_of_bands + 1,
    )


def assign_wavenumber_indices_to_bands(
    effective_window_size: int,
    period_band_edges: np.ndarray,
) -> list[np.ndarray]:
    all_wavenumber_indices = np.arange(1, effective_window_size // 2 + 1)
    all_periods_in_days = effective_window_size / all_wavenumber_indices
    number_of_bands = len(period_band_edges) - 1
    wavenumber_indices_per_band = []
    for band_index in range(number_of_bands):
        lower_period = period_band_edges[band_index]
        upper_period = period_band_edges[band_index + 1]
        if band_index < number_of_bands - 1:
            in_band_mask = (
                (all_periods_in_days >= lower_period)
                & (all_periods_in_days < upper_period)
            )
        else:
            in_band_mask = (
                (all_periods_in_days >= lower_period)
                & (all_periods_in_days <= upper_period)
            )
        wavenumber_indices_per_band.append(all_wavenumber_indices[in_band_mask])
    return wavenumber_indices_per_band


def sum_spectrum_within_bands(
    one_sided_spectrum: np.ndarray,
    wavenumber_indices_per_band: list[np.ndarray],
) -> np.ndarray:
    return np.array([
        one_sided_spectrum[wavenumber_indices - 1].sum()
        if len(wavenumber_indices) > 0 else 0.0
        for wavenumber_indices in wavenumber_indices_per_band
    ])


def compute_rolling_variance_and_spectrum(
    log_returns: pd.Series,
    window_size: int,
    wavenumber_indices_per_band: list[np.ndarray],
    spectrum_estimator: Callable[[pd.Series], np.ndarray],
) -> tuple[pd.Series, pd.DataFrame]:
    output_dates = log_returns.index[window_size - 1:]
    number_of_output_dates = len(output_dates)
    number_of_bands = len(wavenumber_indices_per_band)
    rolling_variances = np.empty(number_of_output_dates)
    rolling_band_variance_matrix = np.empty((number_of_output_dates, number_of_bands))
    for position in range(number_of_output_dates):
        window_returns = log_returns.iloc[position : position + window_size]
        rolling_variances[position] = compute_annualized_variance(window_returns)
        one_sided_spectrum = spectrum_estimator(window_returns)
        rolling_band_variance_matrix[position] = sum_spectrum_within_bands(
            one_sided_spectrum, wavenumber_indices_per_band
        )
    rolling_variance_series = pd.Series(rolling_variances, index=output_dates)
    rolling_band_variance_dataframe = pd.DataFrame(
        rolling_band_variance_matrix, index=output_dates
    )
    return rolling_variance_series, rolling_band_variance_dataframe


def compute_rolling_band_variances_from_window_estimator(
    log_returns: pd.Series,
    window_size: int,
    band_variance_estimator: Callable[[pd.Series], np.ndarray],
) -> tuple[pd.Series, pd.DataFrame]:
    output_dates = log_returns.index[window_size - 1:]
    number_of_output_dates = len(output_dates)
    rolling_variances = np.empty(number_of_output_dates)
    rolling_band_variance_matrix = np.empty(
        (number_of_output_dates, NUMBER_OF_FREQUENCY_BANDS)
    )
    for position in range(number_of_output_dates):
        window_returns = log_returns.iloc[position : position + window_size]
        rolling_variances[position] = compute_annualized_variance(window_returns)
        rolling_band_variance_matrix[position] = band_variance_estimator(window_returns)
    rolling_variance_series = pd.Series(rolling_variances, index=output_dates)
    rolling_band_variance_dataframe = pd.DataFrame(
        rolling_band_variance_matrix, index=output_dates
    )
    return rolling_variance_series, rolling_band_variance_dataframe


def compute_morlet_period_from_scale(scales: np.ndarray) -> np.ndarray:
    omega_0 = MORLET_CENTRAL_FREQUENCY
    return 4.0 * np.pi * scales / (omega_0 + np.sqrt(2.0 + omega_0 ** 2))


def build_morlet_scales_for_period_range(
    minimum_period_in_days: float,
    maximum_period_in_days: float,
    scales_per_octave: int,
) -> np.ndarray:
    omega_0 = MORLET_CENTRAL_FREQUENCY
    period_to_scale_factor = (omega_0 + np.sqrt(2.0 + omega_0 ** 2)) / (4.0 * np.pi)
    minimum_scale = minimum_period_in_days * period_to_scale_factor
    maximum_scale = maximum_period_in_days * period_to_scale_factor
    number_of_scales = (
        int(np.log2(maximum_scale / minimum_scale) * scales_per_octave) + 1
    )
    return minimum_scale * 2.0 ** (np.arange(number_of_scales) / scales_per_octave)


def compute_morlet_wavelet_in_frequency_domain(
    angular_frequencies: np.ndarray,
    scale: float,
) -> np.ndarray:
    omega_0 = MORLET_CENTRAL_FREQUENCY
    normalization = np.sqrt(2.0 * np.pi * scale) * np.pi ** (-0.25)
    wavelet = normalization * np.exp(
        -0.5 * (scale * angular_frequencies - omega_0) ** 2
    )
    wavelet[angular_frequencies < 0] = 0.0
    return wavelet


def compute_cwt_scalogram(
    demeaned_signal: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    number_of_time_points = len(demeaned_signal)
    angular_frequencies = 2.0 * np.pi * np.fft.fftfreq(number_of_time_points, d=1.0)
    signal_fft = np.fft.fft(demeaned_signal)
    scalogram = np.empty((len(scales), number_of_time_points))
    for scale_index, scale in enumerate(scales):
        wavelet_fft = compute_morlet_wavelet_in_frequency_domain(
            angular_frequencies, scale
        )
        coefficients = np.fft.ifft(signal_fft * wavelet_fft)
        scalogram[scale_index] = np.abs(coefficients) ** 2
    return scalogram


def compute_wavelet_variance_normalization_constant(
    scales: np.ndarray,
    delta_j: float,
    number_of_realizations: int = 50,
) -> float:
    rng = np.random.default_rng(seed=42)
    normalization_ratios = []
    for _ in range(number_of_realizations):
        noise = rng.standard_normal(ROLLING_WINDOW_SIZE)
        noise -= noise.mean()
        scalogram = compute_cwt_scalogram(noise, scales)
        wavelet_sum = delta_j * np.sum(scalogram.mean(axis=1) / scales)
        normalization_ratios.append(noise.var() / wavelet_sum)
    return float(np.mean(normalization_ratios))


def assign_scale_indices_to_bands(
    scales: np.ndarray,
    period_band_edges: np.ndarray,
) -> list[np.ndarray]:
    periods = compute_morlet_period_from_scale(scales)
    number_of_bands = len(period_band_edges) - 1
    scale_indices_per_band = []
    for band_index in range(number_of_bands):
        lower_period = period_band_edges[band_index]
        upper_period = period_band_edges[band_index + 1]
        if band_index < number_of_bands - 1:
            in_band_mask = (periods >= lower_period) & (periods < upper_period)
        else:
            in_band_mask = (periods >= lower_period) & (periods <= upper_period)
        scale_indices_per_band.append(np.where(in_band_mask)[0])
    return scale_indices_per_band


def compute_wavelet_band_variances(
    scalogram: np.ndarray,
    scales: np.ndarray,
    period_band_edges: np.ndarray,
    delta_j: float,
    variance_normalization_constant: float,
    log_returns_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    scale_weighted_power = scalogram / scales[:, np.newaxis]
    scale_indices_per_band = assign_scale_indices_to_bands(scales, period_band_edges)
    annualized_normalization = (
        delta_j * variance_normalization_constant * TRADING_DAYS_PER_YEAR
    )
    number_of_bands = len(period_band_edges) - 1
    band_variance_matrix = np.zeros((scalogram.shape[1], number_of_bands))
    for band_index, scale_indices in enumerate(scale_indices_per_band):
        if len(scale_indices) > 0:
            band_variance_matrix[:, band_index] = (
                annualized_normalization
                * scale_weighted_power[scale_indices].sum(axis=0)
            )
    return pd.DataFrame(band_variance_matrix, index=log_returns_index)


def compute_spectral_cdf(band_variances: pd.DataFrame) -> pd.DataFrame:
    total_variance = band_variances.sum(axis=1).replace(0.0, np.nan)
    normalized_pmf = band_variances.div(total_variance, axis=0).fillna(0.0)
    return normalized_pmf.cumsum(axis=1)


def compute_three_band_frequency_masks(
    period_band_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    band_center_periods = np.sqrt(period_band_edges[:-1] * period_band_edges[1:])
    high_frequency_mask = (
        band_center_periods < HIGH_MID_FREQUENCY_BOUNDARY_IN_TRADING_DAYS
    )
    mid_frequency_mask = (
        (band_center_periods >= HIGH_MID_FREQUENCY_BOUNDARY_IN_TRADING_DAYS)
        & (band_center_periods <= MID_LOW_FREQUENCY_BOUNDARY_IN_TRADING_DAYS)
    )
    low_frequency_mask = (
        band_center_periods > MID_LOW_FREQUENCY_BOUNDARY_IN_TRADING_DAYS
    )
    return high_frequency_mask, mid_frequency_mask, low_frequency_mask


def aggregate_to_three_frequency_band_fractions(
    band_variances: pd.DataFrame,
    high_frequency_mask: np.ndarray,
    mid_frequency_mask: np.ndarray,
    low_frequency_mask: np.ndarray,
) -> pd.DataFrame:
    high = band_variances.iloc[:, high_frequency_mask].sum(axis=1)
    mid = band_variances.iloc[:, mid_frequency_mask].sum(axis=1)
    low = band_variances.iloc[:, low_frequency_mask].sum(axis=1)
    total = (high + mid + low).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "high": high / total * 100,
            "mid": mid / total * 100,
            "low": low / total * 100,
        },
        index=band_variances.index,
    )


def compute_single_window_three_band_fractions(
    window_returns: pd.Series,
    band_variance_estimator: Callable[[pd.Series], np.ndarray],
    high_frequency_mask: np.ndarray,
    mid_frequency_mask: np.ndarray,
    low_frequency_mask: np.ndarray,
) -> np.ndarray:
    band_variances = band_variance_estimator(window_returns)
    total = band_variances.sum()
    if total == 0.0:
        return np.array([0.0, 0.0, 0.0])
    return np.array([
        band_variances[high_frequency_mask].sum() / total * 100,
        band_variances[mid_frequency_mask].sum() / total * 100,
        band_variances[low_frequency_mask].sum() / total * 100,
    ])


def compute_white_noise_permutation_bounds(
    band_variance_estimator: Callable[[pd.Series], np.ndarray],
    high_frequency_mask: np.ndarray,
    mid_frequency_mask: np.ndarray,
    low_frequency_mask: np.ndarray,
    window_size: int,
    number_of_permutations: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=42)
    band_fractions = np.empty((number_of_permutations, 3))
    for permutation_index in range(number_of_permutations):
        noise = pd.Series(rng.standard_normal(window_size))
        band_fractions[permutation_index] = compute_single_window_three_band_fractions(
            noise,
            band_variance_estimator,
            high_frequency_mask,
            mid_frequency_mask,
            low_frequency_mask,
        )
    lower_bounds = np.percentile(band_fractions, 2.5, axis=0)
    upper_bounds = np.percentile(band_fractions, 97.5, axis=0)
    return lower_bounds, upper_bounds


def compute_band_variance_correlation_matrix(
    band_variances: pd.DataFrame,
) -> np.ndarray:
    return band_variances.corr().values


def compute_sample_autocorrelation_at_lag(series: np.ndarray, lag: int) -> float:
    number_of_observations = len(series)
    centered = series - series.mean()
    variance = np.mean(centered ** 2)
    if variance == 0.0:
        return 0.0
    return float(
        np.mean(centered[: number_of_observations - lag] * centered[lag:]) / variance
    )


def compute_ljung_box_pvalue(series: np.ndarray, number_of_lags: int) -> float:
    number_of_observations = len(series)
    autocorrelations = np.array([
        compute_sample_autocorrelation_at_lag(series, lag)
        for lag in range(1, number_of_lags + 1)
    ])
    q_statistic = number_of_observations * (number_of_observations + 2) * np.sum(
        autocorrelations ** 2
        / (number_of_observations - np.arange(1, number_of_lags + 1))
    )
    return float(1.0 - stats.chi2.cdf(q_statistic, df=number_of_lags))


def compute_ljung_box_pvalues_per_scale(
    scalogram: np.ndarray,
    number_of_lags: int,
) -> np.ndarray:
    return np.array([
        compute_ljung_box_pvalue(scalogram[scale_index], number_of_lags)
        for scale_index in range(scalogram.shape[0])
    ])


def build_date_edges_for_pcolormesh(date_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    last_step_size = date_index[-1] - date_index[-2]
    return date_index.append(pd.DatetimeIndex([date_index[-1] + last_step_size]))


def configure_heatmap_panel(
    axis: plt.Axes,
    period_band_edges: np.ndarray,
    show_y_labels: bool = True,
    years_per_tick: int = 10,
) -> None:
    axis.set_yscale("log")
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.xaxis.set_major_locator(mdates.YearLocator(years_per_tick))
    plt.setp(axis.xaxis.get_majorticklabels(), rotation=0)
    visible_horizon_ticks = {
        period: label
        for period, label in IMPORTANT_HORIZON_TICKS_IN_TRADING_DAYS.items()
        if period_band_edges[0] <= period <= period_band_edges[-1]
    }
    axis.set_yticks(list(visible_horizon_ticks.keys()))
    if show_y_labels:
        axis.set_yticklabels(list(visible_horizon_ticks.values()))
        axis.set_ylabel("horizon")
    else:
        axis.set_yticklabels([])


def plot_rolling_variance_and_spectrum(
    rolling_variance: pd.Series,
    rolling_band_variances: pd.DataFrame,
    period_band_edges: np.ndarray,
    figure_title: str,
    cone_of_influence_period: float = None,
) -> plt.Figure:
    figure = plt.figure(figsize=(14, 9))
    gridspec = figure.add_gridspec(3, 1, height_ratios=[2, 4, 0.12], hspace=0.35)
    top_axis = figure.add_subplot(gridspec[0])
    bottom_axis = figure.add_subplot(gridspec[1], sharex=top_axis)
    colorbar_axis = figure.add_subplot(gridspec[2])
    figure.suptitle(figure_title)

    top_axis.plot(
        rolling_variance.index, rolling_variance.values,
        color="black", linewidth=0.8, label="variance",
    )
    top_axis.plot(
        rolling_band_variances.index,
        rolling_band_variances.sum(axis=1).values,
        color="red", linewidth=0.8, linestyle="--", alpha=0.7,
        label="TFR implied variance",
    )
    top_axis.legend(loc="upper right", framealpha=0.8)
    top_axis.set_ylabel("annualized variance")
    top_axis.tick_params(labelbottom=False)

    for date_string, label in CRISIS_DATES_AND_LABELS.items():
        crisis_date = pd.Timestamp(date_string)
        if rolling_variance.index.min() <= crisis_date <= rolling_variance.index.max():
            top_axis.axvline(
                crisis_date, color="blue", linewidth=0.8, linestyle="--", alpha=0.7
            )
            top_axis.text(
                crisis_date, top_axis.get_ylim()[1], label,
                rotation=90, fontsize=7, color="blue", alpha=0.85, va="top", ha="right",
            )

    date_edges = build_date_edges_for_pcolormesh(rolling_variance.index)
    color_mesh = bottom_axis.pcolormesh(
        date_edges, period_band_edges, rolling_band_variances.values.T,
        cmap="viridis", shading="flat",
    )
    configure_heatmap_panel(bottom_axis, period_band_edges, show_y_labels=True, years_per_tick=5)

    if cone_of_influence_period is not None:
        bottom_axis.axhline(
            cone_of_influence_period, color="white", linewidth=1.2,
            linestyle="--", alpha=0.85, label="cone of influence boundary",
        )
        bottom_axis.legend(loc="lower right", framealpha=0.6, fontsize=8)

    figure.colorbar(
        color_mesh, cax=colorbar_axis, orientation="horizontal",
        label="annualized variance",
    )
    return figure


def plot_wavelet_variance_and_spectrum(
    rolling_variance: pd.Series,
    wavelet_band_variances: pd.DataFrame,
    period_band_edges: np.ndarray,
    cone_of_influence_period: float,
) -> plt.Figure:
    figure = plt.figure(figsize=(14, 9))
    gridspec = figure.add_gridspec(3, 1, height_ratios=[2, 4, 0.12], hspace=0.35)
    top_axis = figure.add_subplot(gridspec[0])
    bottom_axis = figure.add_subplot(gridspec[1], sharex=top_axis)
    colorbar_axis = figure.add_subplot(gridspec[2])
    figure.suptitle(
        "Annualized variance and variance spectrum, IBOVESPA -- Morlet CWT"
    )

    wavelet_instantaneous_variance = (
        wavelet_band_variances.sum(axis=1)
        .rolling(window=WAVELET_TFR_SMOOTHING_WINDOW, min_periods=1)
        .mean()
    )
    top_axis.plot(
        wavelet_instantaneous_variance.index, wavelet_instantaneous_variance.values,
        color="red", linewidth=0.8, linestyle="--", alpha=0.7,
        label=f"TFR implied variance ({WAVELET_TFR_SMOOTHING_WINDOW}-day smoothed)",
    )
    top_axis.plot(
        rolling_variance.index, rolling_variance.values,
        color="black", linewidth=0.8, label="variance (750-day rolling)",
    )
    top_axis.legend(loc="upper right", framealpha=0.8)
    top_axis.set_ylabel("annualized variance")
    top_axis.tick_params(labelbottom=False)

    for date_string, label in CRISIS_DATES_AND_LABELS.items():
        crisis_date = pd.Timestamp(date_string)
        if wavelet_band_variances.index.min() <= crisis_date <= wavelet_band_variances.index.max():
            top_axis.axvline(
                crisis_date, color="blue", linewidth=0.8, linestyle="--", alpha=0.7
            )
            top_axis.text(
                crisis_date, top_axis.get_ylim()[1], label,
                rotation=90, fontsize=7, color="blue", alpha=0.85, va="top", ha="right",
            )

    date_edges = build_date_edges_for_pcolormesh(wavelet_band_variances.index)
    color_mesh = bottom_axis.pcolormesh(
        date_edges, period_band_edges, wavelet_band_variances.values.T,
        cmap="viridis", shading="flat",
    )
    configure_heatmap_panel(bottom_axis, period_band_edges, show_y_labels=True, years_per_tick=5)
    bottom_axis.axhline(
        cone_of_influence_period, color="white", linewidth=1.2,
        linestyle="--", alpha=0.85, label="cone of influence boundary",
    )
    bottom_axis.legend(loc="lower right", framealpha=0.6, fontsize=8)

    figure.colorbar(
        color_mesh, cax=colorbar_axis, orientation="horizontal",
        label="annualized variance",
    )
    return figure


def plot_spectral_decomposition_with_white_noise_bounds_2x2(
    method_data: list[tuple[str, pd.DataFrame, np.ndarray, np.ndarray]],
) -> plt.Figure:
    band_colors = {"high": "tab:red", "mid": "tab:green", "low": "tab:blue"}
    band_labels = {
        "high": f"high (period < {int(HIGH_MID_FREQUENCY_BOUNDARY_IN_TRADING_DAYS)}d)",
        "mid": (
            f"mid ({int(HIGH_MID_FREQUENCY_BOUNDARY_IN_TRADING_DAYS)}d"
            f" to {int(MID_LOW_FREQUENCY_BOUNDARY_IN_TRADING_DAYS)}d)"
        ),
        "low": f"low (period > {int(MID_LOW_FREQUENCY_BOUNDARY_IN_TRADING_DAYS)}d)",
    }

    figure, axes = plt.subplots(
        2, 2, figsize=(16, 10), sharex=False, sharey=True
    )
    figure.suptitle(
        "Spectral decomposition of rolling variance by frequency band, IBOVESPA\n"
        f"shaded regions: 95% white noise bounds"
        f" ({NUMBER_OF_WHITE_NOISE_PERMUTATIONS:,} permutations)"
    )

    for axis, (method_title, three_band_fractions, lower_bounds, upper_bounds) in zip(
        axes.flatten(), method_data
    ):
        for band_index, band_name in enumerate(["high", "mid", "low"]):
            color = band_colors[band_name]
            series = three_band_fractions[band_name].dropna()
            axis.plot(
                series.index, series.values,
                color=color, linewidth=0.8, label=band_labels[band_name],
            )
            axis.axhspan(
                lower_bounds[band_index], upper_bounds[band_index],
                color=color, alpha=0.12,
            )
            axis.axhline(
                lower_bounds[band_index],
                color=color, linewidth=0.7, linestyle="--", alpha=0.6,
            )
            axis.axhline(
                upper_bounds[band_index],
                color=color, linewidth=0.7, linestyle="--", alpha=0.6,
            )

        axis.set_title(method_title)
        axis.set_ylim(0, 100)
        axis.set_ylabel("percent of total variance (%)")
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.xaxis.set_major_locator(mdates.YearLocator(10))

    axes[0, 0].legend(loc="upper right", framealpha=0.8, fontsize=8)
    figure.tight_layout()
    return figure


def plot_variance_comparison_2x2(
    method_data: list[tuple[str, pd.Series, pd.Series]],
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=False, sharey=True)
    figure.suptitle("Sample variance vs TFR implied variance by method, IBOVESPA")
    for axis, (method_title, rolling_variance, tfr_implied_variance) in zip(
        axes.flatten(), method_data
    ):
        axis.plot(
            rolling_variance.index, rolling_variance.values,
            color="black", linewidth=0.8, label="variance",
        )
        axis.plot(
            tfr_implied_variance.index, tfr_implied_variance.values,
            color="red", linewidth=0.8, linestyle="--", alpha=0.7,
            label="TFR implied variance",
        )
        axis.set_title(method_title)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.xaxis.set_major_locator(mdates.YearLocator(10))
    for axis in axes[:, 0]:
        axis.set_ylabel("annualized variance")
    axes[0, 0].legend(loc="upper right", framealpha=0.8)
    figure.tight_layout()
    return figure


def plot_tfr_comparison_2x2(
    method_data: list[tuple[str, pd.DataFrame]],
    period_band_edges: np.ndarray,
    cone_of_influence_panel_index: int,
    cone_of_influence_period: float,
) -> plt.Figure:
    positive_values = np.concatenate([
        band_variances.values[band_variances.values > 0].ravel()
        for _, band_variances in method_data
    ])
    shared_vmin = float(np.percentile(positive_values, 1))
    shared_vmax = float(np.percentile(positive_values, 99))
    log_norm = mcolors.LogNorm(vmin=shared_vmin, vmax=shared_vmax)

    figure = plt.figure(figsize=(16, 10))
    gridspec = figure.add_gridspec(
        3, 2, height_ratios=[4, 4, 0.12], hspace=0.35, wspace=0.08
    )
    panel_positions = [
        gridspec[0, 0], gridspec[0, 1], gridspec[1, 0], gridspec[1, 1]
    ]
    first_axis = figure.add_subplot(panel_positions[0])
    axes = [first_axis] + [
        figure.add_subplot(position, sharey=first_axis, sharex=first_axis)
        for position in panel_positions[1:]
    ]
    colorbar_axis = figure.add_subplot(gridspec[2, :])
    figure.suptitle("Variance spectrum TFR by method, IBOVESPA (log color scale)")

    last_mesh = None
    for panel_index, (axis, (method_title, band_variances)) in enumerate(
        zip(axes, method_data)
    ):
        date_edges = build_date_edges_for_pcolormesh(band_variances.index)
        last_mesh = axis.pcolormesh(
            date_edges, period_band_edges, band_variances.values.T,
            cmap="viridis", shading="flat", norm=log_norm,
        )
        show_y_labels = panel_index in (0, 2)
        configure_heatmap_panel(axis, period_band_edges, show_y_labels=show_y_labels)
        axis.set_title(method_title)
        if panel_index in (0, 1):
            plt.setp(axis.get_xticklabels(), visible=False)
        if panel_index == cone_of_influence_panel_index:
            axis.axhline(
                cone_of_influence_period, color="white", linewidth=1.2,
                linestyle="--", alpha=0.85, label="COI boundary",
            )
            axis.legend(loc="lower right", framealpha=0.6, fontsize=8)

    figure.colorbar(
        last_mesh, cax=colorbar_axis, orientation="horizontal",
        label="annualized variance (log scale)",
    )
    return figure


def plot_cdf_comparison_2x2(
    method_data: list[tuple[str, pd.DataFrame]],
    period_band_edges: np.ndarray,
    cone_of_influence_panel_index: int,
    cone_of_influence_period: float,
) -> plt.Figure:
    figure = plt.figure(figsize=(16, 10))
    gridspec = figure.add_gridspec(
        3, 2, height_ratios=[4, 4, 0.12], hspace=0.35, wspace=0.08
    )
    panel_positions = [
        gridspec[0, 0], gridspec[0, 1], gridspec[1, 0], gridspec[1, 1]
    ]
    first_axis = figure.add_subplot(panel_positions[0])
    axes = [first_axis] + [
        figure.add_subplot(position, sharey=first_axis, sharex=first_axis)
        for position in panel_positions[1:]
    ]
    colorbar_axis = figure.add_subplot(gridspec[2, :])
    figure.suptitle(
        "Cumulative variance distribution across frequency bands by method, IBOVESPA"
    )

    last_mesh = None
    for panel_index, (axis, (method_title, band_variances)) in enumerate(
        zip(axes, method_data)
    ):
        spectral_cdf = compute_spectral_cdf(band_variances)
        date_edges = build_date_edges_for_pcolormesh(band_variances.index)
        last_mesh = axis.pcolormesh(
            date_edges, period_band_edges, spectral_cdf.values.T,
            cmap="plasma", shading="flat", vmin=0.0, vmax=1.0,
        )
        show_y_labels = panel_index in (0, 2)
        configure_heatmap_panel(axis, period_band_edges, show_y_labels=show_y_labels)
        axis.set_title(method_title)
        if panel_index in (0, 1):
            plt.setp(axis.get_xticklabels(), visible=False)
        if panel_index == cone_of_influence_panel_index:
            axis.axhline(
                cone_of_influence_period, color="white", linewidth=1.2,
                linestyle="--", alpha=0.85, label="COI boundary",
            )
            axis.legend(loc="lower right", framealpha=0.6, fontsize=8)

    figure.colorbar(
        last_mesh, cax=colorbar_axis, orientation="horizontal",
        label="cumulative fraction of variance",
    )
    return figure


period_band_edges = build_log_spaced_period_band_edges(
    ROLLING_WINDOW_SIZE, NUMBER_OF_FREQUENCY_BANDS
)
wavenumber_indices_per_band_full_window = assign_wavenumber_indices_to_bands(
    ROLLING_WINDOW_SIZE, period_band_edges
)
wavenumber_indices_per_band_welch = assign_wavenumber_indices_to_bands(
    WELCH_SUB_SEGMENT_LENGTH, period_band_edges
)

index_log_returns = load_index_log_returns(RAW_DATA_DIRECTORY)

print("computing rectangular window STFT...")
rolling_variance_rectangular, rolling_band_variances_rectangular = (
    compute_rolling_variance_and_spectrum(
        index_log_returns, ROLLING_WINDOW_SIZE,
        wavenumber_indices_per_band_full_window,
        compute_one_sided_annualized_variance_spectrum_rectangular,
    )
)

print("computing Welch STFT...")
rolling_variance_welch, rolling_band_variances_welch = (
    compute_rolling_variance_and_spectrum(
        index_log_returns, ROLLING_WINDOW_SIZE,
        wavenumber_indices_per_band_welch,
        compute_one_sided_annualized_variance_spectrum_welch,
    )
)

print("computing DWT...")
rolling_variance_dwt, rolling_band_variances_dwt = (
    compute_rolling_band_variances_from_window_estimator(
        index_log_returns,
        ROLLING_WINDOW_SIZE,
        lambda window_returns: compute_dwt_band_variances_for_window(
            window_returns, period_band_edges
        ),
    )
)

morlet_scales = build_morlet_scales_for_period_range(
    minimum_period_in_days=2.0,
    maximum_period_in_days=float(ROLLING_WINDOW_SIZE),
    scales_per_octave=MORLET_SCALES_PER_OCTAVE,
)
morlet_delta_j = 1.0 / MORLET_SCALES_PER_OCTAVE

print("computing Morlet CWT normalization constant...")
morlet_normalization_constant = compute_wavelet_variance_normalization_constant(
    morlet_scales, morlet_delta_j
)

print("computing full Morlet CWT scalogram...")
demeaned_log_returns = index_log_returns.values - index_log_returns.values.mean()
full_scalogram = compute_cwt_scalogram(demeaned_log_returns, morlet_scales)

print("computing Morlet wavelet band variances...")
wavelet_band_variances = compute_wavelet_band_variances(
    full_scalogram, morlet_scales, period_band_edges,
    morlet_delta_j, morlet_normalization_constant, index_log_returns.index,
)

cone_of_influence_scale = ROLLING_WINDOW_SIZE / (2.0 * np.sqrt(2.0))
cone_of_influence_period = float(
    compute_morlet_period_from_scale(np.array([cone_of_influence_scale]))[0]
)

comparison_start_date = rolling_variance_rectangular.index[0]
wavelet_band_variances_trimmed = wavelet_band_variances.loc[comparison_start_date:]
wavelet_tfr_implied_smoothed = (
    wavelet_band_variances_trimmed.sum(axis=1)
    .rolling(window=WAVELET_TFR_SMOOTHING_WINDOW, min_periods=1)
    .mean()
)

figure_rectangular = plot_rolling_variance_and_spectrum(
    rolling_variance_rectangular, rolling_band_variances_rectangular,
    period_band_edges,
    "Trailing 750-trading-day annualized variance and variance spectrum, IBOVESPA"
    " -- rectangular window",
)
figure_welch = plot_rolling_variance_and_spectrum(
    rolling_variance_welch, rolling_band_variances_welch,
    period_band_edges,
    "Trailing 750-trading-day annualized variance and variance spectrum, IBOVESPA"
    " -- Welch (sub-segment 375 days, 50% overlap, Hann taper)",
)
figure_dwt = plot_rolling_variance_and_spectrum(
    rolling_variance_dwt, rolling_band_variances_dwt,
    period_band_edges,
    f"Trailing 750-trading-day annualized variance and variance spectrum, IBOVESPA"
    f" -- DWT ({DWT_WAVELET_NAME})",
)
figure_wavelet = plot_wavelet_variance_and_spectrum(
    rolling_variance_rectangular, wavelet_band_variances,
    period_band_edges, cone_of_influence_period,
)

high_frequency_mask, mid_frequency_mask, low_frequency_mask = (
    compute_three_band_frequency_masks(period_band_edges)
)

morlet_scale_indices_per_band = assign_scale_indices_to_bands(
    morlet_scales, period_band_edges
)

def morlet_single_window_band_variance_estimator(
    window_returns: pd.Series,
) -> np.ndarray:
    demeaned = window_returns.values - window_returns.values.mean()
    scalogram = compute_cwt_scalogram(demeaned, morlet_scales)
    scale_weighted_mean_power = scalogram.mean(axis=1) / morlet_scales
    annualized_normalization = (
        morlet_delta_j * morlet_normalization_constant * TRADING_DAYS_PER_YEAR
    )
    band_variances = np.zeros(NUMBER_OF_FREQUENCY_BANDS)
    for band_index, scale_indices in enumerate(morlet_scale_indices_per_band):
        if len(scale_indices) > 0:
            band_variances[band_index] = (
                annualized_normalization * scale_weighted_mean_power[scale_indices].sum()
            )
    return band_variances


rectangular_single_window_estimator = lambda window_returns: sum_spectrum_within_bands(
    compute_one_sided_annualized_variance_spectrum_rectangular(window_returns),
    wavenumber_indices_per_band_full_window,
)
welch_single_window_estimator = lambda window_returns: sum_spectrum_within_bands(
    compute_one_sided_annualized_variance_spectrum_welch(window_returns),
    wavenumber_indices_per_band_welch,
)
dwt_single_window_estimator = lambda window_returns: (
    compute_dwt_band_variances_for_window(window_returns, period_band_edges)
)

print("computing white noise permutation bounds...")
rectangular_lower_bounds, rectangular_upper_bounds = compute_white_noise_permutation_bounds(
    rectangular_single_window_estimator,
    high_frequency_mask, mid_frequency_mask, low_frequency_mask,
    ROLLING_WINDOW_SIZE, NUMBER_OF_WHITE_NOISE_PERMUTATIONS,
)
welch_lower_bounds, welch_upper_bounds = compute_white_noise_permutation_bounds(
    welch_single_window_estimator,
    high_frequency_mask, mid_frequency_mask, low_frequency_mask,
    ROLLING_WINDOW_SIZE, NUMBER_OF_WHITE_NOISE_PERMUTATIONS,
)
dwt_lower_bounds, dwt_upper_bounds = compute_white_noise_permutation_bounds(
    dwt_single_window_estimator,
    high_frequency_mask, mid_frequency_mask, low_frequency_mask,
    ROLLING_WINDOW_SIZE, NUMBER_OF_WHITE_NOISE_PERMUTATIONS,
)
morlet_lower_bounds, morlet_upper_bounds = compute_white_noise_permutation_bounds(
    morlet_single_window_band_variance_estimator,
    high_frequency_mask, mid_frequency_mask, low_frequency_mask,
    ROLLING_WINDOW_SIZE, NUMBER_OF_WHITE_NOISE_PERMUTATIONS,
)

three_band_fractions_rectangular = aggregate_to_three_frequency_band_fractions(
    rolling_band_variances_rectangular,
    high_frequency_mask, mid_frequency_mask, low_frequency_mask,
)
three_band_fractions_welch = aggregate_to_three_frequency_band_fractions(
    rolling_band_variances_welch,
    high_frequency_mask, mid_frequency_mask, low_frequency_mask,
)
three_band_fractions_dwt = aggregate_to_three_frequency_band_fractions(
    rolling_band_variances_dwt,
    high_frequency_mask, mid_frequency_mask, low_frequency_mask,
)
three_band_fractions_morlet = aggregate_to_three_frequency_band_fractions(
    wavelet_band_variances_trimmed,
    high_frequency_mask, mid_frequency_mask, low_frequency_mask,
)

variance_comparison_method_data = [
    (
        "rectangular window",
        rolling_variance_rectangular,
        rolling_band_variances_rectangular.sum(axis=1),
    ),
    (
        "Welch",
        rolling_variance_welch,
        rolling_band_variances_welch.sum(axis=1),
    ),
    (
        f"DWT ({DWT_WAVELET_NAME})",
        rolling_variance_dwt,
        rolling_band_variances_dwt.sum(axis=1),
    ),
    (
        "Morlet CWT",
        rolling_variance_rectangular,
        wavelet_tfr_implied_smoothed,
    ),
]

tfr_and_cdf_method_data = [
    ("rectangular window", rolling_band_variances_rectangular),
    ("Welch", rolling_band_variances_welch),
    (f"DWT ({DWT_WAVELET_NAME})", rolling_band_variances_dwt),
    ("Morlet CWT", wavelet_band_variances_trimmed),
]

spectral_decomposition_method_data = [
    (
        "rectangular window",
        three_band_fractions_rectangular,
        rectangular_lower_bounds,
        rectangular_upper_bounds,
    ),
    (
        "Welch",
        three_band_fractions_welch,
        welch_lower_bounds,
        welch_upper_bounds,
    ),
    (
        f"DWT ({DWT_WAVELET_NAME})",
        three_band_fractions_dwt,
        dwt_lower_bounds,
        dwt_upper_bounds,
    ),
    (
        "Morlet CWT",
        three_band_fractions_morlet,
        morlet_lower_bounds,
        morlet_upper_bounds,
    ),
]

figure_variance_comparison = plot_variance_comparison_2x2(variance_comparison_method_data)

figure_diagnostics = plot_spectral_decomposition_with_white_noise_bounds_2x2(
    spectral_decomposition_method_data,
)

figure_tfr_comparison = plot_tfr_comparison_2x2(
    tfr_and_cdf_method_data, period_band_edges,
    cone_of_influence_panel_index=3,
    cone_of_influence_period=cone_of_influence_period,
)

figure_cdf_comparison = plot_cdf_comparison_2x2(
    tfr_and_cdf_method_data, period_band_edges,
    cone_of_influence_panel_index=3,
    cone_of_influence_period=cone_of_influence_period,
)

plt.show()