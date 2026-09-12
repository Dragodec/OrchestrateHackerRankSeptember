from collections import deque

import pandas as pd


class CurrencyConverter:
    """
    Convert amounts between dataset currencies.

    Uses exchange rates from the CSV and can traverse indirect
    conversion paths when a direct pair is unavailable.

    Example:
        INR -> USD -> EUR -> ZAR

    Reverse directions are derived from the supplied dataset rate:
        if USD -> INR = rate,
        then INR -> USD = 1 / rate.
    """

    def __init__(self, exchange_rates):
        self.exchange_rates = exchange_rates.copy()

        self.exchange_rates["rate_date"] = pd.to_datetime(
            self.exchange_rates["rate_date"]
        ).dt.normalize()

        self.exchange_rates = (
            self.exchange_rates
            .sort_values("rate_date")
            .reset_index(drop=True)
        )

    def convert(
        self,
        amount,
        from_currency,
        to_currency,
        date,
    ):
        """
        Convert an amount into another currency.
        """

        amount = float(amount)

        if from_currency == to_currency:
            return amount

        rate = self.get_rate(
            from_currency=from_currency,
            to_currency=to_currency,
            date=date,
        )

        if rate is None:
            raise ValueError(
                f"No conversion path available for "
                f"{from_currency} -> {to_currency} "
                f"on {date}"
            )

        return amount * rate

    def get_rate(
        self,
        from_currency,
        to_currency,
        date,
    ):
        """
        Find a conversion rate using direct, inverse,
        or multi-hop dataset paths.
        """

        date = pd.Timestamp(date).normalize()

        graph = self._build_graph(date)

        return self._find_path_rate(
            graph,
            from_currency,
            to_currency,
        )

    def _build_graph(self, date):
        """
        Build a graph of the latest available rates on or
        before the requested date.
        """

        valid_rates = self.exchange_rates[
            self.exchange_rates["rate_date"] <= date
        ]

        graph = {}

        for (
            from_currency,
            to_currency,
        ), group in valid_rates.groupby(
            ["from_currency", "to_currency"]
        ):

            latest = (
                group
                .sort_values("rate_date")
                .iloc[-1]
            )

            rate = float(latest["rate"])

            graph.setdefault(
                from_currency,
                {}
            )[to_currency] = rate

            # Reverse conversion derived from the same
            # dataset-provided rate.
            if rate != 0:
                graph.setdefault(
                    to_currency,
                    {}
                )[from_currency] = 1 / rate

        return graph

    @staticmethod
    def _find_path_rate(
        graph,
        from_currency,
        to_currency,
    ):
        """
        Find a conversion path with BFS and multiply rates
        along that path.
        """

        queue = deque(
            [(from_currency, 1.0)]
        )

        visited = {from_currency}

        while queue:

            currency, accumulated_rate = queue.popleft()

            if currency == to_currency:
                return accumulated_rate

            for next_currency, rate in graph.get(
                currency,
                {},
            ).items():

                if next_currency in visited:
                    continue

                visited.add(next_currency)

                queue.append(
                    (
                        next_currency,
                        accumulated_rate * rate,
                    )
                )

        return None