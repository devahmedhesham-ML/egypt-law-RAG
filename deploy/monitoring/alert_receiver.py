"""Logs every Grafana alert notification (webhook contact point): one line per alert, firing or resolved."""

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        for a in body.get("alerts", []):
            labels, ann = a.get("labels", {}), a.get("annotations", {})
            print(f"[{now}] {a.get('status', '?').upper()}: {labels.get('alertname')} "
                  f"(run={labels.get('label', '-')}) {ann.get('summary', '')}", flush=True)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # only the alert lines
        pass


HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
