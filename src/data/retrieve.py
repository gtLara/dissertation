import logging
from datetime import date
from pathlib import Path

import yaml

from src.data.providers.yfinance_provider import YFinanceProvider


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PARAMETERS_PATH = Path("params.yaml")
RAW_DATA_DIRECTORY = Path("data/raw")


def load_retrieval_parameters() -> dict:
    """
    Load data retrieval parameters from the project parameter file.

    Reads ``params.yaml`` at the project root and returns the nested
    dictionary under the ``retrieval`` key.

    :returns: A dictionary with keys ``index_ticker``,
        ``constituent_ticker``, ``start_date``, and optionally
        ``end_date``.
    :rtype: dict
    :raises KeyError: If the ``retrieval`` section is absent from
        ``params.yaml``.
    """
    with open(PARAMETERS_PATH) as parameter_file:
        return yaml.safe_load(parameter_file)["retrieval"]


def resolve_end_date(raw_end_date: str | None) -> date:
    """
    Resolve the end date from a raw string parameter.

    If the parameter is the string ``"today"`` or is absent, the
    current date is used. Otherwise the string is parsed as an ISO 8601
    date.

    :param raw_end_date: The raw end date string from the parameter
        file, or ``None``.
    :type raw_end_date: str or None
    :returns: The resolved end date.
    :rtype: date
    """
    if raw_end_date is None or raw_end_date == "today":
        return date.today()
    return date.fromisoformat(raw_end_date)


def fetch_and_persist(
    provider: YFinanceProvider,
    ticker: str,
    start_date: date,
    end_date: date,
    output_filename: str,
) -> None:
    """
    Fetch price data for a single ticker and write it to Parquet.

    :param provider: An initialised market data provider.
    :type provider: MarketDataProvider
    :param ticker: The ticker symbol to retrieve.
    :type ticker: str
    :param start_date: The start of the requested date range, inclusive.
    :type start_date: date
    :param end_date: The end of the requested date range, inclusive.
    :type end_date: date
    :param output_filename: The filename of the output Parquet file,
        written into ``data/raw/``.
    :type output_filename: str
    """
    logger.info("Fetching %s from %s to %s", ticker, start_date, end_date)
    data = provider.fetch(ticker, start_date, end_date)

    actual_start = data.index.min().date()
    actual_end = data.index.max().date()
    logger.info(
        "Retrieved %d trading days for %s  [%s -> %s]",
        len(data),
        ticker,
        actual_start,
        actual_end,
    )

    output_path = RAW_DATA_DIRECTORY / output_filename
    data.to_parquet(output_path)
    logger.info("Saved to %s", output_path)


def main() -> None:
    """
    Entry point for the data retrieval pipeline stage.

    Reads ticker symbols and date range from ``params.yaml``, downloads
    the index and constituent series via the configured provider, and
    persists each as a Parquet file under ``data/raw/``.
    """
    parameters = load_retrieval_parameters()
    provider = YFinanceProvider()

    start_date = date.fromisoformat(parameters["start_date"])
    end_date = resolve_end_date(parameters.get("end_date"))

    logger.info(
        "Provider: %s", type(provider).__name__
    )
    logger.info(
        "Requested date range: %s to %s", start_date, end_date
    )

    fetch_and_persist(
        provider=provider,
        ticker=parameters["index_ticker"],
        start_date=start_date,
        end_date=end_date,
        output_filename="index.parquet",
    )

    fetch_and_persist(
        provider=provider,
        ticker=parameters["constituent_ticker"],
        start_date=start_date,
        end_date=end_date,
        output_filename="constituent.parquet",
    )

    logger.info("Retrieval complete.")


if __name__ == "__main__":
    main()