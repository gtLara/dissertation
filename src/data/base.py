from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class MarketDataProvider(ABC):
    """
    Abstract base class for market data providers.

    Defines the interface for retrieving historical price data.
    Concrete subclasses wrap specific data sources such as remote
    financial APIs or local databases, and must implement
    :meth:`fetch`. The :meth:`fetch_multiple` method has a default
    implementation that calls :meth:`fetch` in a loop; subclasses may
    override it to exploit provider-level batch endpoints.
    """

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch historical daily price data for a single ticker.

        :param ticker: The ticker symbol as recognised by the data provider.
        :type ticker: str
        :param start_date: The first date of the requested data range,
            inclusive.
        :type start_date: date
        :param end_date: The last date of the requested data range,
            inclusive.
        :type end_date: date
        :returns: A DataFrame indexed by date with at minimum a
            ``close`` column. Column names must be lowercase.
        :rtype: pandas.DataFrame
        :raises ValueError: If the provider returns no data for the
            requested ticker and date range.
        """

    def fetch_multiple(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical daily price data for multiple tickers.

        The default implementation delegates to :meth:`fetch` for each
        ticker. Subclasses may override this method to use a more
        efficient batch request when the underlying provider supports it.

        :param tickers: A list of ticker symbols recognised by the
            provider.
        :type tickers: list[str]
        :param start_date: The first date of the requested data range,
            inclusive.
        :type start_date: date
        :param end_date: The last date of the requested data range,
            inclusive.
        :type end_date: date
        :returns: A dictionary mapping each ticker to its corresponding
            price DataFrame, as returned by :meth:`fetch`.
        :rtype: dict[str, pandas.DataFrame]
        """
        return {ticker: self.fetch(ticker, start_date, end_date) for ticker in tickers}
