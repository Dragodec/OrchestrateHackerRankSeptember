from collections import defaultdict
from datetime import timedelta

import pandas as pd


class ForecastEngine:
    """
    Simulate a user's cash balance over a 90-day horizon.

    Forecast sources:
    - recurring patterns detected from settled history
    - scheduled financial events already known in the dataset
    - hypothetical payments supplied by the planner

    All cash flows are converted into the user's home currency.
    """

    def __init__(
        self,
        events,
        recurrence_detector,
        currency_converter,
        horizon_days=90,
    ):
        self.events = events.copy()
        self.recurrence_detector = recurrence_detector
        self.currency_converter = currency_converter
        self.horizon_days = horizon_days

    def forecast(
        self,
        user_id,
        profile,
        as_of_date,
        payments=None,
    ):
        """
        Run a forecast and return daily balances.

        payments should be a list of dictionaries:

        {
            "date": Timestamp,
            "amount": float,
            "currency": "USD",
            "direction": "debit"
        }

        If currency is omitted, the user's home currency is assumed.
        """

        as_of_date = pd.Timestamp(as_of_date).normalize()

        end_date = (
            as_of_date
            + timedelta(days=self.horizon_days)
        )

        home_currency = profile["home_currency"]

        cash_flows = defaultdict(float)

        # --------------------------------------------------
        # Known future scheduled events
        # --------------------------------------------------

        self._add_scheduled_events(
            cash_flows=cash_flows,
            user_id=user_id,
            home_currency=home_currency,
            start_date=as_of_date,
            end_date=end_date,
        )

        # --------------------------------------------------
        # Recurring patterns from historical evidence
        # --------------------------------------------------

        patterns = self.recurrence_detector.detect(
            user_id=user_id,
            as_of_date=as_of_date,
        )

        self._add_recurring_patterns(
            cash_flows=cash_flows,
            patterns=patterns,
            home_currency=home_currency,
            start_date=as_of_date,
            end_date=end_date,
        )

        # --------------------------------------------------
        # Hypothetical request payments
        # --------------------------------------------------

        if payments:

            self._add_payments(
                cash_flows=cash_flows,
                payments=payments,
                home_currency=home_currency,
                start_date=as_of_date,
                end_date=end_date,
            )

        # --------------------------------------------------
        # Daily simulation
        # --------------------------------------------------

        starting_balance = float(
            profile["current_available_balance"]
        )

        minimum_balance = float(
            profile["minimum_balance_to_keep"]
        )

        balance = starting_balance

        rows = []

        current_date = as_of_date

        while current_date <= end_date:

            net_change = cash_flows.get(
                current_date,
                0.0,
            )

            balance += net_change

            rows.append(
                {
                    "date": current_date,
                    "net_change": net_change,
                    "balance": balance,
                    "minimum_balance": minimum_balance,
                }
            )

            current_date += timedelta(days=1)

        forecast_df = pd.DataFrame(rows)

        return {
            "forecast": forecast_df,
            "patterns": patterns,
            "minimum_balance": minimum_balance,
            "starting_balance": starting_balance,
            "home_currency": home_currency,
        }

    def is_safe(
        self,
        user_id,
        profile,
        as_of_date,
        payments=None,
    ):
        """
        Check whether the forecast ever drops below
        minimum_balance_to_keep.
        """

        result = self.forecast(
            user_id=user_id,
            profile=profile,
            as_of_date=as_of_date,
            payments=payments,
        )

        forecast = result["forecast"]

        minimum_balance = result["minimum_balance"]

        safe = (
            forecast["balance"] >= minimum_balance
        ).all()

        lowest_balance = float(
            forecast["balance"].min()
        )

        return {
            "safe": bool(safe),
            "lowest_balance": lowest_balance,
            "result": result,
        }

    # ==================================================
    # CASH FLOW SOURCES
    # ==================================================

    def _add_scheduled_events(
        self,
        cash_flows,
        user_id,
        home_currency,
        start_date,
        end_date,
    ):
        """
        Add confirmed future scheduled cash events.

        Pending events are excluded.
        """

        events = self.events[
            (self.events["user_id"] == user_id)
            & (self.events["status"] == "scheduled")
            & (self.events["direction"].isin(["debit", "credit"]))
            & (self.events["amount"].notna())
        ]

        for _, event in events.iterrows():

            event_date = pd.Timestamp(
                event["event_date"]
            ).normalize()

            if not (
                start_date <= event_date <= end_date
            ):
                continue

            amount = self.currency_converter.convert(
                amount=event["amount"],
                from_currency=event["currency"],
                to_currency=home_currency,
                date=event_date,
            )

            signed_amount = self._signed_amount(
                amount,
                event["direction"],
            )

            cash_flows[event_date] += signed_amount

    def _add_recurring_patterns(
        self,
        cash_flows,
        patterns,
        home_currency,
        start_date,
        end_date,
    ):
        """
        Extend detected recurring patterns through
        the forecast horizon.
        """

        for pattern in patterns:

            next_date = pd.Timestamp(
                pattern["last_date"]
            ).normalize()

            interval_days = int(
                pattern["interval_days"]
            )

            # Move forward until inside forecast window.
            while next_date <= start_date:
                next_date += timedelta(
                    days=interval_days
                )

            while next_date <= end_date:

                amount = self.currency_converter.convert(
                    amount=pattern["amount"],
                    from_currency=pattern["currency"],
                    to_currency=home_currency,
                    date=next_date,
                )

                signed_amount = self._signed_amount(
                    amount,
                    pattern["direction"],
                )

                cash_flows[next_date] += signed_amount

                next_date += timedelta(
                    days=interval_days
                )

    def _add_payments(
        self,
        cash_flows,
        payments,
        home_currency,
        start_date,
        end_date,
    ):
        """
        Add hypothetical payments being evaluated.
        """

        for payment in payments:

            payment_date = pd.Timestamp(
                payment["date"]
            ).normalize()

            if not (
                start_date <= payment_date <= end_date
            ):
                continue

            payment_currency = payment.get(
                "currency",
                home_currency,
            )

            amount = self.currency_converter.convert(
                amount=payment["amount"],
                from_currency=payment_currency,
                to_currency=home_currency,
                date=payment_date,
            )

            direction = payment.get(
                "direction",
                "debit",
            )

            signed_amount = self._signed_amount(
                amount,
                direction,
            )

            cash_flows[payment_date] += signed_amount

    @staticmethod
    def _signed_amount(amount, direction):

        if direction == "credit":
            return float(amount)

        if direction == "debit":
            return -float(amount)

        return 0.0