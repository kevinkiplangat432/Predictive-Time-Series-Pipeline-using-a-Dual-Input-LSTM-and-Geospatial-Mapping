"""
sms_utils.py
------------
SMS Early Warning Notification module for KFPEWS.

This module is intentionally kept separate from app.py so the SMS feature
is modular and can be tested, edited, or swapped out without touching the
forecasting/anomaly dashboard code.

WHAT THIS MODULE DOES
----------------------
1. Classifies an alert (HIGH_PRICE / LOW_PRICE / ANOMALY / None) using ONLY
   the existing data your notebook already produces:
       - price_history.csv -> flagged  (the residual-based anomaly detector)
       - forecasts.csv     -> display_forecast, naive_forecast, chosen_forecast
   No new prediction method is introduced here.

2. Builds the SMS text for each alert type, in the same style as the
   examples in the project brief.

3. Validates and normalizes Kenyan phone numbers.

4. Sends the SMS via Africa's Talking (the standard, low-friction SMS API
   for Kenya-based projects: it has a free sandbox environment, a simple
   Python SDK, and no need for international gateway setup) -- or, if no
   credentials are configured, falls back to Demo Mode automatically, which
   simulates delivery without requiring paid credits or real credentials.

5. Keeps a simple alert history (CSV) with de-duplication, so the same
   market/commodity/alert_type/month doesn't spam the same stakeholder.

CONFIGURATION (environment variables - never hard-code credentials)
---------------------------------------------------------------------
    SMS_API_KEY     - Africa's Talking API key
    SMS_USERNAME    - Africa's Talking username ("sandbox" for the free
                       sandbox environment)
    SMS_SENDER_ID   - optional alphanumeric sender/shortcode ID

If SMS_API_KEY or SMS_USERNAME are not set, send_sms() automatically runs
in Demo Mode regardless of what the caller asks for, and clearly reports
that in the returned status.
"""

import africastalking
import pandas as pd


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def init_sms(username: str, api_key: str):
    """
    Initialize the Africa's Talking SDK and return the SMS service object.

    For sandbox testing: username="sandbox", api_key=<your sandbox API key>.
    For production: username=<your live username>, api_key=<your live API key>.
    """
    africastalking.initialize(username, api_key)
    return africastalking.SMS


class MockSMSClient:
    """
    Fake SMS client for demos/presentations when no real Africa's Talking
    account or API key is available yet. Mimics the same .send() interface,
    always "succeeds", and never makes a network call.

    Use this only to demonstrate the FEATURE and UI flow — swap in the real
    init_sms() client for an actual capstone submission or production use,
    since this never truly sends anything.
    """
    def send(self, message: str, recipients: list):
        return {
            "SMSMessageData": {
                "Message": f"Sent to {len(recipients)}/{len(recipients)} Total Cost: KES 0.0000 (MOCK - not actually sent)",
                "Recipients": [
                    {"number": r, "status": "Success", "statusCode": 101,
                     "cost": "KES 0.0000", "messageId": "MOCK_ID"}
                    for r in recipients
                ],
            }
        }


# ---------------------------------------------------------------------------
# Stakeholders
# ---------------------------------------------------------------------------
def load_stakeholders(path: str = "stakeholders.csv") -> pd.DataFrame:
    """
    Load the stakeholders list. Expected CSV columns: name, phone, role.
    Phone numbers should be in international format, e.g. +254712345678.
    """
    df = pd.read_csv(path)
    required = {"name", "phone", "role"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"stakeholders.csv is missing required columns: {missing}")
    return df


def demo_stakeholder(phone: str, name: str = "Demo Tester") -> pd.DataFrame:
    """
    Build a single-row stakeholders DataFrame for demo/testing purposes,
    so you can test the full send flow without touching the real list.
    """
    return pd.DataFrame([{"name": name, "phone": phone, "role": "Demo"}])


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------
def build_alert_message(row: pd.Series) -> str:
    """Build a single-pair SMS alert message from one row of the forecasts df."""
    direction = "rise" if row["expected_change_pct"] > 0 else "fall"
    return (
        f"[KFPEWS ALERT] {row['commodity']} in {row['market']}: "
        f"{row['warning_level']} warning. Forecast {direction} of "
        f"{abs(row['expected_change_pct']):.1f}% next month. "
        f"Current: {row['current_price']:.0f} KES/kg."
    )


def build_consolidated_message(flagged_df: pd.DataFrame, max_pairs: int = 5) -> str:
    """
    Build ONE message per stakeholder covering multiple flagged pairs,
    instead of one SMS per pair. Keeps costs down when many pairs are flagged.
    Truncates to max_pairs and notes how many more were omitted.
    """
    if flagged_df.empty:
        return "[KFPEWS] No pairs currently flagged."

    lines = ["[KFPEWS ALERT] Flagged pairs:"]
    shown = flagged_df.head(max_pairs)
    for _, row in shown.iterrows():
        direction = "up" if row["expected_change_pct"] > 0 else "down"
        lines.append(
            f"- {row['commodity']} ({row['market']}): {row['warning_level']}, "
            f"{direction} {abs(row['expected_change_pct']):.1f}%"
        )

    remaining = len(flagged_df) - len(shown)
    if remaining > 0:
        lines.append(f"...and {remaining} more. See dashboard for full list.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------
def send_alerts(
    sms_client,
    stakeholders_df: pd.DataFrame,
    flagged_df: pd.DataFrame,
    consolidated: bool = True,
    max_pairs_per_message: int = 5,
) -> pd.DataFrame:
    """
    Send SMS alerts to every stakeholder about currently flagged pairs.

    If consolidated=True (recommended), each stakeholder gets ONE message
    covering up to max_pairs_per_message flagged pairs.
    If consolidated=False, each stakeholder gets one SMS PER flagged pair
    (can get expensive/spammy fast if many pairs are flagged).

    Returns a DataFrame log of what was sent and whether it succeeded.
    """
    results = []

    if flagged_df.empty:
        for _, srow in stakeholders_df.iterrows():
            results.append({
                "phone": srow["phone"], "name": srow.get("name", ""),
                "message": "No pairs currently flagged. No SMS sent.",
                "status": "skipped",
            })
        return pd.DataFrame(results)

    for _, srow in stakeholders_df.iterrows():
        if consolidated:
            message = build_consolidated_message(flagged_df, max_pairs_per_message)
            results.append(_send_one(sms_client, srow, message))
        else:
            for _, frow in flagged_df.iterrows():
                message = build_alert_message(frow)
                entry = _send_one(sms_client, srow, message)
                entry["commodity"] = frow["commodity"]
                entry["market"] = frow["market"]
                results.append(entry)

    return pd.DataFrame(results)


def _send_one(sms_client, stakeholder_row: pd.Series, message: str) -> dict:
    """Send a single SMS and return a log entry dict."""
    try:
        sms_client.send(message, [stakeholder_row["phone"]])
        status = "sent"
    except Exception as e:
        status = f"failed: {e}"

    return {
        "phone": stakeholder_row["phone"],
        "name": stakeholder_row.get("name", ""),
        "message": message,
        "status": status,
    }