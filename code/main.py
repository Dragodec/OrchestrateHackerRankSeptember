from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).parent))

from data.loader import DataLoader


def main():

    dataset_dir = Path(__file__).parent.parent / "dataset"
    project_root = Path(__file__).parent.parent

    loader = DataLoader(dataset_dir)
    datasets = loader.load_all()

    events = datasets["financial_events"]
    messages = datasets["messages"]

    sample_requests = pd.read_csv(
        dataset_dir / "sample_requests.csv"
    )

    # Ensure date columns are usable.
    for column in [
        "event_date",
        "settlement_date",
    ]:
        if column in events.columns:
            events[column] = pd.to_datetime(
                events[column],
                errors="coerce",
            )

    if "sent_at" in messages.columns:
        messages["sent_at"] = pd.to_datetime(
            messages["sent_at"],
            errors="coerce",
        )

    sample_requests["request_date"] = pd.to_datetime(
        sample_requests["request_date"],
        errors="coerce",
    )

    output_lines = []

    def log(text=""):

        text = str(text)

        print(text)
        output_lines.append(text)

    # ============================================================
    # HEADER
    # ============================================================

    log("=" * 80)
    log("SAMPLE USER FUTURE FINANCIAL INFORMATION DIAGNOSTIC")
    log("=" * 80)

    sample_user_ids = set(
        sample_requests["user_id"].dropna()
    )

    sample_request_ids = set(
        sample_requests["request_id"].dropna()
    )

    log()
    log(f"Sample requests: {len(sample_requests)}")
    log(f"Unique sample users: {len(sample_user_ids)}")

    # ============================================================
    # GLOBAL EVENT STATUS DIAGNOSTIC
    # ============================================================

    log()
    log("=" * 80)
    log("GLOBAL FUTURE EVENT STATUS")
    log("=" * 80)

    future_events = events[
        events["event_date"].notna()
    ]

    log()
    log(
        future_events["status"]
        .value_counts()
        .to_string()
    )

    # ============================================================
    # REQUEST-BY-REQUEST ANALYSIS
    # ============================================================

    summary_rows = []

    for _, request in sample_requests.iterrows():

        request_id = request["request_id"]
        user_id = request["user_id"]

        request_date = pd.Timestamp(
            request["request_date"]
        ).normalize()

        horizon_end = (
            request_date
            + pd.Timedelta(days=90)
        )

        user_events = events[
            events["user_id"] == user_id
        ].copy()

        # --------------------------------------------------------
        # FUTURE EVENTS IN 90-DAY WINDOW
        # --------------------------------------------------------

        future_window_events = user_events[
            user_events["event_date"].between(
                request_date,
                horizon_end,
                inclusive="both",
            )
        ].copy()

        # --------------------------------------------------------
        # SCHEDULED EVENTS
        # --------------------------------------------------------

        scheduled_events = future_window_events[
            future_window_events["status"]
            == "scheduled"
        ].copy()

        # --------------------------------------------------------
        # FUTURE PENDING EVENTS
        # --------------------------------------------------------

        pending_events = future_window_events[
            future_window_events["status"]
            == "pending"
        ].copy()

        # --------------------------------------------------------
        # FUTURE EVENTS BY DIRECTION
        # --------------------------------------------------------

        future_credits = future_window_events[
            future_window_events["direction"]
            == "credit"
        ]

        future_debits = future_window_events[
            future_window_events["direction"]
            == "debit"
        ]

        # --------------------------------------------------------
        # EVENT DATE VS SETTLEMENT DATE
        # --------------------------------------------------------

        dated_events = future_window_events[
            future_window_events["settlement_date"].notna()
        ].copy()

        different_dates = dated_events[
            dated_events["event_date"]
            != dated_events["settlement_date"]
        ]

        # --------------------------------------------------------
        # MESSAGES
        #
        # Include:
        # - messages directly belonging to the user
        # - messages directly linked to this request
        # --------------------------------------------------------

        user_messages = messages[
            (
                messages["user_id"] == user_id
            )
            | (
                messages["request_id"] == request_id
            )
        ].copy()

        # ========================================================
        # REPORT
        # ========================================================

        log()
        log()
        log("=" * 80)
        log(f"REQUEST: {request_id}")
        log(f"USER: {user_id}")
        log("=" * 80)

        log(
            f"Request date: "
            f"{request_date.date()}"
        )

        log(
            f"90-day horizon ends: "
            f"{horizon_end.date()}"
        )

        # --------------------------------------------------------

        log()
        log("--- FUTURE EVENTS IN WINDOW ---")

        log(
            f"Total: "
            f"{len(future_window_events)}"
        )

        log(
            f"Credits: "
            f"{len(future_credits)}"
        )

        log(
            f"Debits: "
            f"{len(future_debits)}"
        )

        if not future_window_events.empty:

            log()

            display_columns = [
                "event_id",
                "event_type",
                "description",
                "category",
                "direction",
                "amount",
                "currency",
                "event_date",
                "settlement_date",
                "status",
                "flexibility",
                "linked_event_id",
            ]

            available_columns = [
                column
                for column in display_columns
                if column in future_window_events.columns
            ]

            log(
                future_window_events[
                    available_columns
                ].sort_values(
                    "event_date"
                ).to_string(
                    index=False
                )
            )

        # --------------------------------------------------------

        log()
        log("--- SCHEDULED EVENTS ---")

        log(
            f"Count: "
            f"{len(scheduled_events)}"
        )

        if not scheduled_events.empty:

            log(
                scheduled_events[
                    [
                        "event_id",
                        "category",
                        "direction",
                        "amount",
                        "currency",
                        "event_date",
                        "settlement_date",
                    ]
                ].sort_values(
                    "event_date"
                ).to_string(
                    index=False
                )
            )

        # --------------------------------------------------------

        log()
        log("--- PENDING EVENTS ---")

        log(
            f"Count: "
            f"{len(pending_events)}"
        )

        if not pending_events.empty:

            log(
                pending_events[
                    [
                        "event_id",
                        "category",
                        "direction",
                        "amount",
                        "currency",
                        "event_date",
                        "settlement_date",
                    ]
                ].sort_values(
                    "event_date"
                ).to_string(
                    index=False
                )
            )

        # --------------------------------------------------------

        log()
        log("--- EVENT DATE / SETTLEMENT DATE DIFFERENCES ---")

        log(
            f"Count: "
            f"{len(different_dates)}"
        )

        if not different_dates.empty:

            log(
                different_dates[
                    [
                        "event_id",
                        "category",
                        "direction",
                        "event_date",
                        "settlement_date",
                        "status",
                    ]
                ].sort_values(
                    "event_date"
                ).to_string(
                    index=False
                )
            )

        # --------------------------------------------------------

        log()
        log("--- RELATED MESSAGES ---")

        log(
            f"Count: "
            f"{len(user_messages)}"
        )

        if not user_messages.empty:

            for _, message in user_messages.sort_values(
                "sent_at"
            ).iterrows():

                log()

                log(
                    f"[{message['sent_at']}] "
                    f"{message['source_type']}"
                )

                log(
                    f"related_event_id: "
                    f"{message['related_event_id']}"
                )

                log(
                    f"request_id: "
                    f"{message['request_id']}"
                )

                log(
                    f"MESSAGE: "
                    f"{message['message_text']}"
                )

        # ========================================================
        # SUMMARY ROW
        # ========================================================

        summary_rows.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "request_date": request_date.date(),
                "future_events": len(
                    future_window_events
                ),
                "scheduled": len(
                    scheduled_events
                ),
                "pending": len(
                    pending_events
                ),
                "future_credits": len(
                    future_credits
                ),
                "future_debits": len(
                    future_debits
                ),
                "date_settlement_differences": len(
                    different_dates
                ),
                "messages": len(
                    user_messages
                ),
            }
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    log()
    log()
    log("=" * 80)
    log("SUMMARY")
    log("=" * 80)

    summary_df = pd.DataFrame(
        summary_rows
    )

    log()
    log(
        summary_df.to_string(
            index=False
        )
    )

    # ============================================================
    # WRITE OUTPUT
    # ============================================================

    output_path = (
        project_root / "terminalOutput.txt"
    )

    output_path.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(
        f"Diagnostic written to: "
        f"{output_path}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()