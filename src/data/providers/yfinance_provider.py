from datetime import date

import pandas as pd
import yfinance as yf

from src.data.base import MarketDataProvider


class YFinanceProvider(MarketDataProvider):
    """
    Market data provider backed by the Yahoo Finance API via yfinance.

    This implementation is intended for rapid prototyping. Data quality,
    availability, and API stability may vary. The provider should be
    replaced with a more reliable source for production use.

    .. note::
        Prices are adjusted for splits and dividends by default via
        ``auto_adjust=True`` in yfinance. Disable this behaviour by
        subclassing and overriding :meth:`fetch` if unadjusted prices
        are required.
    """

    def fetch(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch adjusted daily OHLCV data for a single ticker from Yahoo
        Finance.

        :param ticker: A Yahoo Finance ticker symbol, e.g. ``^BVSP``
            for the Bovespa index or ``VALE3.SA`` for Vale on B3.
        :type ticker: str
        :param start_date: The first date of the requested data range,
            inclusive.
        :type start_date: date
        :param end_date: The last date of the requested data range,
            inclusive.
        :type end_date: date
        :returns: A DataFrame indexed by date with lowercase columns:
            ``open``, ``high``, ``low``, ``close``, ``volume``.
        :rtype: pandas.DataFrame
        :raises ValueError: If Yahoo Finance returns no data for the
            requested ticker and date range.
        """
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )

        if raw.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}' between "
                f"{start_date} and {end_date}."
            )

        raw.columns = [column.lower() for column in raw.columns]
        raw.index.name = "date"
        raw.index = pd.to_datetime(raw.index)
        return raw

    def fetch_multiple(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch adjusted daily OHLCV data for multiple tickers in a single
        Yahoo Finance request.

        Overrides the default loop-based implementation in
        :class:`~src.data.base.MarketDataProvider` to use a single
        batched download.

        :param tickers: A list of Yahoo Finance ticker symbols.
        :type tickers: list[str]
        :param start_date: The first date of the requested data range,
            inclusive.
        :type start_date: date
        :param end_date: The last date of the requested data range,
            inclusive.
        :type end_date: date
        :returns: A dictionary mapping each ticker to its price
            DataFrame, with lowercase columns and a DatetimeIndex named
            ``date``.
        :rtype: dict[str, pandas.DataFrame]
        :raises ValueError: If Yahoo Finance returns no data for the
            requested tickers and date range.
        """
        raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        if raw.empty:
            raise ValueError(
                f"No data returned for tickers {tickers} between "
                f"{start_date} and {end_date}."
            )

        raw.index.name = "date"
        raw.index = pd.to_datetime(raw.index)

        result = {}
        for ticker in tickers:
            ticker_data = raw[ticker].copy()
            ticker_data.columns = [
                column.lower() for column in ticker_data.columns
            ]
            result[ticker] = ticker_data

        return result