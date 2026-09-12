from pathlib import Path

import pandas as pd


class DataLoader:
    """
    Loads all datasets required by the production solver.

    sample_requests.csv is intentionally excluded because it is used
    separately for development and evaluation.
    """

    DATASETS = {
        "requests": "requests.csv",
        "financial_profiles": "financial_profiles.csv",
        "financial_events": "financial_events.csv",
        "exchange_rates": "exchange_rates.csv",
        "request_payment_options": "request_payment_options.csv",
        "messages": "messages.csv",
        "images": "images.csv",
    }

    DATE_COLUMNS = {
        "requests": [
            "request_date",
            "desired_completion_date",
        ],
        "financial_events": [
            "event_date",
            "settlement_date",
        ],
        "exchange_rates": [
            "rate_date",
        ],
        "request_payment_options": [
            "first_payment_date",
        ],
    }

    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.datasets = {}

    def load_all(self):
        """
        Load all production datasets into memory.
        """

        for dataset_name, filename in self.DATASETS.items():

            path = self.dataset_dir / filename

            if not path.exists():
                raise FileNotFoundError(
                    f"Required dataset not found: {path}"
                )

            dataframe = pd.read_csv(path)

            dataframe = self._parse_dates(
                dataset_name,
                dataframe,
            )

            self.datasets[dataset_name] = dataframe

        self._parse_message_timestamps()

        return self.datasets

    def _parse_dates(self, dataset_name, dataframe):
        """
        Parse verified date columns for a dataset.
        """

        for column in self.DATE_COLUMNS.get(dataset_name, []):

            dataframe[column] = pd.to_datetime(
                dataframe[column],
                errors="coerce",
            )

        return dataframe

    def _parse_message_timestamps(self):
        """
        Parse message timestamps as UTC-aware datetimes.
        """

        messages = self.datasets["messages"]

        messages["sent_at"] = pd.to_datetime(
            messages["sent_at"],
            errors="coerce",
            utc=True,
        )

    def get(self, dataset_name):
        """
        Return a loaded dataset.
        """

        if dataset_name not in self.datasets:
            raise KeyError(
                f"Dataset not loaded: {dataset_name}"
            )

        return self.datasets[dataset_name]