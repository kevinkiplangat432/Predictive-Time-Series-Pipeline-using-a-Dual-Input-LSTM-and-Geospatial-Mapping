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
   PythoSDKn , and no need for international gateway setup) -- or, if no
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

import os
import re
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

# --------------------------------------------------------------------------
# CONFIG - thresholds reuse your existing forecast/anomaly columns only.
# Edit these two numbers to tune sensitivity; nothing else needs to change.
# --------------------------------------------------------------------------

# Used when a month IS already flagged by your anomaly detector, to decide
# whether the movement is clearly upward, clearly downward, or too mixed to
# call a direction (in which case it's reported as a plain ANOMALY alert).
ANOMALY_DIRECTION_THRESHOLD_PCT = 3.0

# Used when a month is NOT flagged by the anomaly detector, but the blended
# forecast alone still implies a large swing. This threshold is set higher
# than the anomaly-direction one on purpose: your notebook's own evaluation
# found no model reliably beats naive per pair, so a forecast-only alert
# should require a bigger signal before it's worth messaging a stakeholder.
FORECAST_ONLY_THRESHOLD_PCT = 8.0

ALERT_HISTORY_FILE = os.path.join("data", "alert_history.csv")
ALERT_HISTORY_COLUMNS = [
    "timestamp_sent", "as_of_date", "market", "commodity",
    "alert_type", "basis", "phone_masked", "message", "status", "demo_mode",
]

# --------------------------------------------------------------------------
# ALERT CLASSIFICATION
# --------------------------------------------------------------------------


@dataclass
class AlertDecision:
    alert_type: Optional[str]   # "HIGH_PRICE" | "LOW_PRICE" | "ANOMALY" | None
    basis: Optional[str]        # "anomaly" | "forecast_only" | None
    reason: str                 # human-readable explanation for the UI


def classify_alert(current_price: float, forecast_price: float, flagged: bool) -> AlertDecision:
    """
    Decide what (if any) SMS alert applies, using only the existing
    anomaly flag and blended forecast - the same signals already shown
    on the dashboard. The anomaly flag is treated as the PRIMARY signal;
    the forecast is SUPPORTING context, matching the existing dashboard
    logic.
    """
    if current_price in (None, 0) or current_price != current_price:  # NaN-safe
        return AlertDecision(None, None, "No current price available for this pair.")

    pct_change = (forecast_price - current_price) / current_price * 100

    if flagged:
        # Anomaly-confirmed: the residual-based detector already flagged this
        # month. Use the forecast direction only to decide the wording.
        if pct_change >= ANOMALY_DIRECTION_THRESHOLD_PCT:
            return AlertDecision(
                "HIGH_PRICE", "anomaly",
                f"Anomaly flagged AND forecast direction is upward ({pct_change:+.1f}%).",
            )
        elif pct_change <= -ANOMALY_DIRECTION_THRESHOLD_PCT:
            return AlertDecision(
                "LOW_PRICE", "anomaly",
                f"Anomaly flagged AND forecast direction is downward ({pct_change:+.1f}%).",
            )
        else:
            return AlertDecision(
                "ANOMALY", "anomaly",
                f"Anomaly flagged but forecast direction is unclear ({pct_change:+.1f}%, "
                f"below the {ANOMALY_DIRECTION_THRESHOLD_PCT}% direction threshold).",
            )

    # Not flagged - only alert if the forecast swing alone is large enough
    # to be worth a message. This is explicitly weaker evidence than an
    # anomaly-confirmed alert, and is labelled as such wherever it's shown.
    if pct_change >= FORECAST_ONLY_THRESHOLD_PCT:
        return AlertDecision(
            "HIGH_PRICE", "forecast_only",
            f"No anomaly flagged, but blended forecast implies a large increase ({pct_change:+.1f}%).",
        )
    elif pct_change <= -FORECAST_ONLY_THRESHOLD_PCT:
        return AlertDecision(
            "LOW_PRICE", "forecast_only",
            f"No anomaly flagged, but blended forecast implies a large decrease ({pct_change:+.1f}%).",
        )

    return AlertDecision(None, None, f"No significant movement detected ({pct_change:+.1f}%).")


# --------------------------------------------------------------------------
# SMS MESSAGE BUILDERS
# --------------------------------------------------------------------------


def build_sms_message(alert: AlertDecision, market: str, commodity: str,
                       current_price: float, forecast_price: float) -> str:
    pct_change = (forecast_price - current_price) / current_price * 100 if current_price else 0

    if alert.alert_type == "HIGH_PRICE":
        confidence_note = (
            "" if alert.basis == "anomaly" else
            " (forecast-based signal; this system's forecasts do not reliably "
            "beat a simple previous-price estimate for every market, so treat "
            "this as directional context)"
        )
        return (
            f"KFPEWS ALERT: {commodity} price in {market} has increased significantly"
            f"{confidence_note}. Current price: KES {current_price:.0f}/kg. "
            f"Expected next-month price: KES {forecast_price:.0f}/kg ({pct_change:+.1f}%). "
            f"Stakeholders are advised to monitor the market and review "
            f"purchasing/selling decisions."
        )

    if alert.alert_type == "LOW_PRICE":
        confidence_note = (
            "" if alert.basis == "anomaly" else
            " (forecast-based signal; treat as directional context, not a precise prediction)"
        )
        return (
            f"KFPEWS ALERT: {commodity} price in {market} has decreased significantly"
            f"{confidence_note}. Current price: KES {current_price:.0f}/kg. "
            f"Expected next-month price: KES {forecast_price:.0f}/kg ({pct_change:+.1f}%). "
            f"Farmers and traders should review selling, storage and purchasing decisions."
        )

    if alert.alert_type == "ANOMALY":
        return (
            f"KFPEWS WARNING: An unusual {commodity} price movement has been detected "
            f"in {market}. Current price: KES {current_price:.0f}/kg. The movement "
            f"exceeds the market's historical volatility threshold. Please verify "
            f"local market conditions."
        )

    return (
        f"KFPEWS DEMO: This is a sample alert for {commodity} in {market}. "
        f"Current price: KES {current_price:.0f}/kg. No significant movement is "
        f"currently detected for this pair."
    )


# --------------------------------------------------------------------------
# PHONE NUMBER VALIDATION (Kenyan numbers)
# --------------------------------------------------------------------------

_KENYA_LOCAL = re.compile(r"^0(7|1)\d{8}$")          # e.g. 0712345678
_KENYA_INTL_PLUS = re.compile(r"^\+254(7|1)\d{8}$")   # e.g. +254712345678
_KENYA_INTL_NOPLUS = re.compile(r"^254(7|1)\d{8}$")   # e.g. 254712345678


def normalize_kenyan_phone(raw: str) -> Optional[str]:
    """
    Validate and normalize a Kenyan phone number to E.164 (+254XXXXXXXXX).
    Returns None if the number doesn't match a recognized Kenyan format.
    Accepts Safaricom/Airtel/Telkom style numbers starting 07 or 01.
    """
    if not raw:
        return None
    cleaned = re.sub(r"[\s\-()]", "", raw.strip())

    if _KENYA_LOCAL.match(cleaned):
        return "+254" + cleaned[1:]
    if _KENYA_INTL_PLUS.match(cleaned):
        return cleaned
    if _KENYA_INTL_NOPLUS.match(cleaned):
        return "+" + cleaned
    return None


def mask_phone(phone: str) -> str:
    """Mask the middle digits of a phone number for display, e.g. +254712***678."""
    if not phone or len(phone) < 8:
        return "***"
    return phone[:7] + "***" + phone[-3:]


# --------------------------------------------------------------------------
# SMS SENDING (Africa's Talking, with automatic Demo Mode fallback)
# --------------------------------------------------------------------------


def _has_real_credentials() -> bool:
    return bool(os.environ.get("SMS_API_KEY")) and bool(os.environ.get("SMS_USERNAME"))


def send_sms(phone: str, message: str, demo_mode: bool = True) -> str:
    """
    Send an SMS via Africa's Talking, or simulate sending it in Demo Mode.

    - If demo_mode=True (or no real credentials are configured), no network
      call is made and a simulated "Sent" status is returned instantly.
    - If demo_mode=False AND real credentials ARE configured, this attempts
      a real send through Africa's Talking's Python SDK.

    Returns a short human-readable status string.
    """
    if demo_mode or not _has_real_credentials():
        reason = "" if demo_mode else " (no SMS_API_KEY/SMS_USERNAME configured, auto-fallback)"
        return f"Sent (Demo Mode{reason}) - not actually delivered"

    try:
        import africastalking  # pip install africastalking

        africastalking.initialize(
            username=os.environ["SMS_USERNAME"],
            api_key=os.environ["SMS_API_KEY"],
        )
        sms = africastalking.SMS
        sender_id = os.environ.get("SMS_SENDER_ID") or None
        response = sms.send(message, [phone], sender_id=sender_id)
        recipients = response.get("SMSMessageData", {}).get("Recipients", [])
        if recipients and recipients[0].get("status") == "Success":
            return f"Sent (Africa's Talking, status: {recipients[0].get('status')})"
        return f"Failed: {response}"
    except ImportError:
        return "Failed: 'africastalking' package not installed (pip install africastalking)"
    except Exception as exc:  # noqa: BLE001 - surface any provider error to the UI
        return f"Failed: {exc}"


# --------------------------------------------------------------------------
# ALERT HISTORY (CSV-backed, with de-duplication)
# --------------------------------------------------------------------------


def _ensure_history_file():
    os.makedirs(os.path.dirname(ALERT_HISTORY_FILE), exist_ok=True)
    if not os.path.exists(ALERT_HISTORY_FILE):
        with open(ALERT_HISTORY_FILE, "w", newline="") as f:
            csv.writer(f).writerow(ALERT_HISTORY_COLUMNS)


def load_alert_history():
    import pandas as pd
    _ensure_history_file()
    return pd.read_csv(ALERT_HISTORY_FILE)


def already_alerted(history_df, market: str, commodity: str, alert_type: str, as_of_date: str) -> bool:
    """
    Prevents sending a duplicate alert for the same market/commodity/alert
    type/month. as_of_date should be the latest observed date for that pair
    (not today's date), since the alert is about that month's data.
    """
    if history_df.empty:
        return False
    match = history_df[
        (history_df["market"] == market)
        & (history_df["commodity"] == commodity)
        & (history_df["alert_type"] == alert_type)
        & (history_df["as_of_date"] == as_of_date)
        & (history_df["status"].str.startswith("Sent"))
    ]
    return not match.empty


def log_alert(as_of_date: str, market: str, commodity: str, alert_type: str,
              basis: Optional[str], phone: str, message: str, status: str, demo_mode: bool):
    _ensure_history_file()
    with open(ALERT_HISTORY_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            as_of_date, market, commodity, alert_type or "DEMO", basis or "-",
            mask_phone(phone), message, status, demo_mode,
        ])
