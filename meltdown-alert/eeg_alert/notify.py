"""Alert delivery: console, generic webhook, WhatsApp (Twilio).

Credentials are read from environment variables only; nothing is stored in code.
"""
from __future__ import annotations

import json
import os
from typing import List, Protocol

from .detector import Alert


class Notifier(Protocol):
    def send(self, alert: Alert) -> None: ...


def _fmt(alert: Alert) -> str:
    mm, ss = divmod(int(alert.t), 60)
    head = {"EARLY_WARNING": "EARLY WARNING", "IMMINENT": "IMMINENT",
            "CLINICAL_RISK": "CLINICAL NOTICE"}.get(alert.level, alert.level)
    if alert.eta_min is None or alert.level != "EARLY_WARNING":
        eta = ""
    elif alert.eta_min < 1:
        eta = " | ETA <1 min"
    else:
        eta = f" | ETA ~{alert.eta_min:.0f} min"
    return (f"[{mm:02d}:{ss:02d}] {head} | index {alert.index:+.2f} z | "
            f"trend {alert.slope_per_min:+.2f} z/min{eta}\n  {alert.message}")


class ConsoleNotifier:
    def send(self, alert: Alert) -> None:
        print(_fmt(alert), flush=True)


class WebhookNotifier:
    """POST the alert as JSON to any URL (Home Assistant, IFTTT, Slack, a phone app...)."""

    def __init__(self, url: str) -> None:
        self.url = url

    def send(self, alert: Alert) -> None:
        import requests
        payload = {"level": alert.level, "t_sec": alert.t, "index": alert.index,
                   "slope_per_min": alert.slope_per_min, "eta_min": alert.eta_min,
                   "message": alert.message}
        try:
            requests.post(self.url, data=json.dumps(payload),
                          headers={"Content-Type": "application/json"}, timeout=5)
        except Exception as e:  # never crash the monitoring loop
            print(f"[webhook] failed: {e}")


class TwilioWhatsAppNotifier:
    """Send alerts to a caregiver's WhatsApp via the Twilio API.

    Required environment variables:
      TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
      TWILIO_WHATSAPP_FROM  (e.g. whatsapp:+14155238886)
      ALERT_WHATSAPP_TO     (e.g. whatsapp:+9725XXXXXXXX)
    """

    def __init__(self) -> None:
        missing = [k for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
                               "TWILIO_WHATSAPP_FROM", "ALERT_WHATSAPP_TO") if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"Missing environment variables for WhatsApp alerts: {', '.join(missing)}")
        self.sid = os.environ["TWILIO_ACCOUNT_SID"]
        self.token = os.environ["TWILIO_AUTH_TOKEN"]
        self.frm = os.environ["TWILIO_WHATSAPP_FROM"]
        self.to = os.environ["ALERT_WHATSAPP_TO"]

    def send(self, alert: Alert) -> None:
        import requests
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages.json"
        try:
            requests.post(url, auth=(self.sid, self.token),
                          data={"From": self.frm, "To": self.to, "Body": _fmt(alert)}, timeout=10)
        except Exception as e:
            print(f"[whatsapp] failed: {e}")


def build(kinds: List[str], webhook_url: str | None = None) -> List[Notifier]:
    out: List[Notifier] = []
    for k in kinds:
        if k == "console":
            out.append(ConsoleNotifier())
        elif k == "webhook":
            if not webhook_url:
                raise SystemExit("--webhook-url is required for webhook alerts")
            out.append(WebhookNotifier(webhook_url))
        elif k == "whatsapp":
            out.append(TwilioWhatsAppNotifier())
        else:
            raise SystemExit(f"Unknown notifier: {k}")
    return out
