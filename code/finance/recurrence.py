from collections import defaultdict

import pandas as pd


class RecurrenceDetector:
    """
    Detect likely recurring cash-flow patterns from recent settled history.

    This is intentionally conservative:
    - only settled cash events
    - only events before the request date
    - requires at least 3 occurrences
    - focuses on weekly and monthly-like patterns
    """

    def __init__(self, events):
        self.events = events.copy()

    def detect(self, user_id, as_of_date):
        """
        Return likely recurring patterns for a user as of a request date.
        """

        history = self._get_history(user_id, as_of_date)

        groups = self._group_events(history)

        patterns = []

        for _, group in groups.items():
            pattern = self._build_pattern(group, as_of_date)

            if pattern is not None:
                patterns.append(pattern)

        return patterns

    def _get_history(self, user_id, as_of_date):
        """
        Get valid historical cash events available at the request date.
        """

        history = self.events[
            (self.events["user_id"] == user_id)
            & (self.events["status"] == "settled")
            & (self.events["direction"].isin(["debit", "credit"]))
            & (self.events["amount"].notna())
            & (self.events["event_date"] <= as_of_date)
        ].copy()

        return history.sort_values("event_date")

    def _group_events(self, events):
        """
        Group by financial behavior rather than exact amount.

        Amounts can naturally vary for recurring spending such
        as groceries or transport.
        """

        groups = defaultdict(list)

        grouping_columns = [
            "category",
            "direction",
            "currency",
        ]

        for _, event in events.iterrows():

            key = tuple(
                event[column]
                for column in grouping_columns
            )

            groups[key].append(event.to_dict())

        return groups

    def _build_pattern(self, events, as_of_date):
        """
        Determine whether a sequence looks weekly or monthly recurring.
        """

        if len(events) < 3:
            return None

        events = sorted(
            events,
            key=lambda event: event["event_date"],
        )

        # Use the most recent events as the strongest evidence.
        recent_events = events[-6:]

        dates = [
            event["event_date"]
            for event in recent_events
        ]

        intervals = []

        for previous, current in zip(dates, dates[1:]):

            gap = (current - previous).days

            if gap > 0:
                intervals.append(gap)

        if len(intervals) < 2:
            return None

        median_interval = pd.Series(intervals).median()

        frequency = self._classify_frequency(
            median_interval
        )

        if frequency is None:
            return None

        # Recurring patterns should have reasonably consistent gaps.
        interval_spread = max(intervals) - min(intervals)

        if frequency == "weekly":
            max_spread = 7
        else:
            max_spread = 10

        if interval_spread > max_spread:
            return None

        latest_event = recent_events[-1]

        # Don't continue patterns indefinitely after they appear inactive.
        days_since_last = (
            as_of_date - latest_event["event_date"]
        ).days

        if frequency == "weekly":
            max_staleness = 21
            interval_days = 7
        else:
            max_staleness = 62
            interval_days = 30

        if days_since_last > max_staleness:
            return None

        amounts = [
            event["amount"]
            for event in recent_events
        ]

        # Median is more robust than copying one unusual transaction.
        amount = float(pd.Series(amounts).median())

        return {
            "user_id": latest_event["user_id"],
            "category": latest_event["category"],
            "direction": latest_event["direction"],
            "currency": latest_event["currency"],
            "amount": amount,
            "frequency": frequency,
            "interval_days": interval_days,
            "last_date": latest_event["event_date"],
            "event_count": len(recent_events),
            "flexibility": self._resolve_flexibility(
                recent_events
            ),
            "source_event_ids": [
                event["event_id"]
                for event in recent_events
            ],
        }

    @staticmethod
    def _classify_frequency(median_interval):
        """
        Classify only the recurrence types useful for our
        first baseline forecast.
        """

        if 5 <= median_interval <= 9:
            return "weekly"

        if 25 <= median_interval <= 35:
            return "monthly"

        return None

    @staticmethod
    def _resolve_flexibility(events):
        """
        Use the latest event's flexibility.

        Later evidence should generally represent the most
        current classification of that recurring expense.
        """

        return events[-1]["flexibility"]