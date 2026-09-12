from collections import defaultdict
import pandas as pd


def has_value(value):
    """
    True when a value is genuinely present.

    Used only for deciding whether a record can be added to a
    particular relationship index. The underlying record is never
    discarded because another field is missing.
    """

    return pd.notna(value)


class Resolver:
    """
    Builds fast lookup indices for relationships between datasets.

    All records remain available even when some fields are blank.
    A missing relationship key only prevents indexing that record
    under that specific relationship.
    """

    def __init__(self, datasets):

        self.datasets = datasets

        # One profile per user
        self.profiles_by_user = {}

        # One user can have many events
        self.events_by_user = defaultdict(list)

        # One request can have many payment options
        self.payment_options_by_request = defaultdict(list)

        # Messages can be related at different levels
        self.messages_by_user = defaultdict(list)
        self.messages_by_request = defaultdict(list)
        self.messages_by_event = defaultdict(list)

        # Images may provide evidence for a financial event
        self.images_by_event = defaultdict(list)

        self._build_indices()

    # --------------------------------------------------
    # Index construction
    # --------------------------------------------------

    def _build_indices(self):

        self._index_profiles()
        self._index_events()
        self._index_payment_options()
        self._index_messages()
        self._index_images()

    # --------------------------------------------------
    # Financial profiles
    # --------------------------------------------------

    def _index_profiles(self):

        dataframe = self.datasets["financial_profiles"]

        for _, row in dataframe.iterrows():

            profile = row.to_dict()
            user_id = profile.get("user_id")

            if has_value(user_id):
                self.profiles_by_user[user_id] = profile

    # --------------------------------------------------
    # Financial events
    # --------------------------------------------------

    def _index_events(self):

        dataframe = self.datasets["financial_events"]

        for _, row in dataframe.iterrows():

            event = row.to_dict()
            user_id = event.get("user_id")

            # Keep the event intact, including blank fields
            if has_value(user_id):
                self.events_by_user[user_id].append(event)

    # --------------------------------------------------
    # Payment options
    # --------------------------------------------------

    def _index_payment_options(self):

        dataframe = self.datasets[
            "request_payment_options"
        ]

        for _, row in dataframe.iterrows():

            option = row.to_dict()
            request_id = option.get("request_id")

            if has_value(request_id):
                self.payment_options_by_request[
                    request_id
                ].append(option)

    # --------------------------------------------------
    # Messages
    # --------------------------------------------------

    def _index_messages(self):

        dataframe = self.datasets["messages"]

        for _, row in dataframe.iterrows():

            message = row.to_dict()

            user_id = message.get("user_id")
            request_id = message.get("request_id")
            event_id = message.get("related_event_id")

            # A message can belong to multiple contexts.
            # Missing one relationship does not discard it.

            if has_value(user_id):
                self.messages_by_user[user_id].append(
                    message
                )

            if has_value(request_id):
                self.messages_by_request[request_id].append(
                    message
                )

            if has_value(event_id):
                self.messages_by_event[event_id].append(
                    message
                )

    # --------------------------------------------------
    # Images
    # --------------------------------------------------

    def _index_images(self):

        dataframe = self.datasets["images"]

        for _, row in dataframe.iterrows():

            image = row.to_dict()

            event_id = image.get("related_event_id")

            # Keep image records intact. Only the event relationship
            # requires a present event_id for this particular index.

            if has_value(event_id):
                self.images_by_event[event_id].append(
                    image
                )

    # --------------------------------------------------
    # Public lookup API
    # --------------------------------------------------

    def profile(self, user_id):

        return self.profiles_by_user.get(user_id)

    def events(self, user_id):

        return self.events_by_user.get(user_id, [])

    def payment_options(self, request_id):

        return self.payment_options_by_request.get(
            request_id,
            [],
        )

    def user_messages(self, user_id):

        return self.messages_by_user.get(
            user_id,
            [],
        )

    def request_messages(self, request_id):

        return self.messages_by_request.get(
            request_id,
            [],
        )

    def event_messages(self, event_id):

        return self.messages_by_event.get(
            event_id,
            [],
        )

    def event_images(self, event_id):

        return self.images_by_event.get(
            event_id,
            [],
        )