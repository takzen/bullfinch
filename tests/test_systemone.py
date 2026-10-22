import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from bullfinch.questions import TypedQuestion
from bullfinch.systemone import SystemOneEngine


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        q = body["questions"]["q"]
        answer = {"noul": 0.8} if q["type"] == "noul" else {"probabilities": {k: 0.5 for k in q["criteria"]}}
        out = json.dumps({"answers": {"q": answer}, "seen_auth": self.headers.get("Authorization")}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args) -> None:
        pass


def test_systemone_client_maps_noul_and_choice() -> None:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        engine = SystemOneEngine(f"http://127.0.0.1:{server.server_port}", key="k")
        noul = engine.logprobs("s", TypedQuestion("noul", "Czy?", {"true": "tak", "false": "nie"}))
        assert math.exp(noul["true"]) == 0.8 or abs(math.exp(noul["true"]) - 0.8) < 1e-9
        choice = engine.logprobs("s", TypedQuestion("choice", "Który?", {"a": "x", "b": "y"}))
        assert abs(math.exp(choice["a"]) - 0.5) < 1e-9
    finally:
        server.shutdown()
