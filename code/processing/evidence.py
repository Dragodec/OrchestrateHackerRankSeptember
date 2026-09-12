import pandas as pd


class EvidenceProcessor:
    """
    Collects evidence related to financial events.

    This layer links raw events with messages and images but does not
    yet make unsupported financial interpretations.
    """

    def __init__(self, resolver):
        self.resolver = resolver

    def events_for_user(self, user_id):
        """
        Return copies of all financial events for a user, enriched with
        related evidence references.
        """

        raw_events = self.resolver.events(user_id)

        resolved_events = []

        for raw_event in raw_events:

            event = raw_event.copy()

            event_id = event["event_id"]

            messages = self.resolver.event_messages(event_id)
            images = self.resolver.event_images(event_id)

            event["related_messages"] = messages
            event["related_images"] = images

            event["amount_missing"] = pd.isna(
                event.get("amount")
            )

            resolved_events.append(event)

        return resolved_events

    def missing_amount_events(self, user_id):
        """
        Return events whose amounts are missing.

        Missing amounts are preserved as unresolved evidence and must
        not be treated as zero.
        """

        events = self.events_for_user(user_id)

        return [
            event
            for event in events
            if event["amount_missing"]
        ]