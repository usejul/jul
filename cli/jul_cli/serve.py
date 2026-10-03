"""A tiny local HTTP server around `TypeSafeClient` that speaks the Jev (System One) HTTP protocol.

    jul serve --model minicpm5-2b --port 8577

The wire format is TypeSafe's: any Jev client (the official SDKs, or plain `curl`) talks to it by
changing only its base URL.

    POST /v1/systemone
    Authorization: Bearer <key>          (only when a key is configured)
    {
      "model": "jev-latest",             # optional here; a jev-* name means the server's model
      "state": "a string" | {...} | [...],
      "questions": {
        "name": {"type": "choice", "instructions": "...", "criteria": {"key": "description" | null}},
        "name": {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},
        "name": {"type": "score",  "instructions": "...", "criteria": ["lowest", "...", "highest"]}
      }
    }

    -> {"request_id": ..., "model": ..., "usage": {...},
        "answers": {"name": {"type": "choice", "choice": ..., "confidence": ..., "probabilities": {...}}}}

    GET /v1/models   -> {"models": [{"name": ..., "description": ..., "release_date": ...}, ...]}

JuL additions, all optional and ignored by Jev clients:

- request: `context` (name of a saved context), `method`, `route_above`, as in `system_one`;
  a Choice's `criteria` may also be a plain list of keys, as in the library.
- response: `jul: {"latency_ms": ...}`, plus `jul.escalation` (which tier answered each question)
  when the server escalates (`--escalate-to`).
- routes: `GET /health`; `POST /v1/classify` is an alias of `/v1/systemone`.

Binds to 127.0.0.1 by default (local only). Bound elsewhere with `--host`, it should be given a key
(`--api-key` or `$JUL_API_KEY`); requests then need `Authorization: Bearer <key>` (as Jev) or
`x-api-key: <key>`. Standard library only, so `pip install jul` is enough.

The request `state` is never logged: only the question count and the latency are.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger("jul.serve")

#: One client for the whole process, loaded on the first call (or at startup unless --no-warmup).
#: The backends hold a single model in memory, so the client is a shared, guarded singleton.
_client = None
_client_lock = threading.Lock()
#: Inference is serialised: one model, one forward pass at a time.
_infer_lock = threading.Lock()
_default_model: str | None = None
_default_backend: str | None = None
#: When set, every request must carry it. None means no authentication (safe only on loopback).
_api_key: str | None = None
#: Optional escalation: unsure answers go to another System One server (see jul/escalate.py).
_escalate: dict | None = None

MAX_BODY = 10_000_000
SYSTEMONE_PATHS = ("/v1/systemone", "/v1/classify", "/classify")
EXTRAS = ("context", "method", "route_above")


class RequestError(Exception):
    """A request the protocol refuses: `status` and `kind` follow the Jev API."""

    def __init__(self, status: int, kind: str, message: str, detail: list | None = None):
        super().__init__(message)
        self.status, self.kind, self.message, self.detail = status, kind, message, detail

    def payload(self) -> dict:
        if self.detail is not None:  # 422: the list of fields, as a validation error
            return {"detail": self.detail}
        return {"error": {"type": self.kind, "message": self.message}}


def get_client():
    """Get or build the shared client. Thread-safe: the HTTP server is threaded."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            from jul import Choice, TypeSafeClient
            logger.info("loading JuL client (model=%s, backend=%s)", _default_model, _default_backend)
            start = time.time()
            client = TypeSafeClient(model=_default_model, backend=_default_backend)
            # Forces the weights to load now, so the first real request is not the slow one.
            client.system_one(state="warmup",
                              questions={"q": Choice(instructions="test?", criteria={"a": "a", "b": "b"})})
            if _escalate:
                from jul.escalate import Escalation, SystemOneHTTP
                remote = SystemOneHTTP(_escalate["url"], model=_escalate["model"], api_key=_escalate["api_key"])
                client = Escalation([("local", client), ("remote", remote)],
                                    min_confidence=_escalate["min_confidence"])
                logger.info("escalating answers below %.2f to %s (%s)", _escalate["min_confidence"],
                            remote.url, _escalate["model"])
            _client = client
            logger.info("JuL client ready in %.1fs", time.time() - start)
    return _client


def _missing(field: str, where: tuple = ("body",)) -> dict:
    return {"loc": [*where, field], "msg": "Field required", "type": "missing"}


def _text(value: Any) -> str:
    """A description as the prompt reads it: null is empty (the key speaks), JSON stays JSON."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def build_questions(questions_data: Any) -> dict:
    """JuL question objects from the Jev request `questions` map."""
    from jul import Choice, Noul, NoulCriteria, Score

    if not isinstance(questions_data, dict) or not questions_data:
        raise RequestError(422, "invalid_request", "questions must be a non-empty map",
                           detail=[_missing("questions")])
    questions: dict[str, Any] = {}
    for name, q in questions_data.items():
        if not isinstance(q, dict):
            raise RequestError(400, "api_usage_error", f"question {name!r} must be an object")
        kind = str(q.get("type", "")).lower()
        instructions = _text(q.get("instructions", ""))
        criteria = q.get("criteria")
        if kind == "noul":
            if isinstance(criteria, dict):
                questions[name] = Noul(instructions=instructions,
                                       criteria=NoulCriteria(true=_text(criteria.get("true")),
                                                             false=_text(criteria.get("false"))))
            else:
                questions[name] = Noul(instructions=instructions)
        elif kind == "choice":
            if criteria is None:
                raise RequestError(422, "invalid_request", f"question {name!r}: a choice needs criteria",
                                   detail=[_missing("criteria", ("body", "questions", name))])
            if isinstance(criteria, dict):
                criteria = {str(k): _text(v) for k, v in criteria.items()}
            elif isinstance(criteria, list):  # JuL addition: a plain list of keys
                criteria = [str(k) for k in criteria]
            else:
                raise RequestError(400, "api_usage_error", f"question {name!r}: criteria must be a map")
            questions[name] = Choice(instructions=instructions, criteria=criteria)
        elif kind == "score":
            if not isinstance(criteria, list):
                raise RequestError(422, "invalid_request",
                                   f"question {name!r}: a score needs an ordered list of levels",
                                   detail=[_missing("criteria", ("body", "questions", name))])
            questions[name] = Score(instructions=instructions, criteria=[_text(c) for c in criteria])
        else:
            raise RequestError(400, "api_usage_error",
                               f"Invalid request. Unknown question type {kind!r} for {name!r} "
                               "(choice, noul or score)")
    return questions


def local_model(requested: Any) -> str | None:
    """The model to run: a `jev-*` name (what a Jev client sends) means the server's own model."""
    if requested is None or requested == "":
        return None
    if not isinstance(requested, str):
        raise RequestError(400, "api_usage_error", "model must be a string")
    return None if requested.lower().startswith("jev-") else requested


def classify(body: Any) -> dict:
    """One /v1/systemone call: the Jev request in, the Jev response (plus `jul`) out."""
    if not isinstance(body, dict):
        raise RequestError(400, "api_usage_error", "Invalid request. The body must be a JSON object")
    if "state" not in body:
        raise RequestError(422, "invalid_request", "state is required", detail=[_missing("state")])
    questions = build_questions(body.get("questions"))
    model = local_model(body.get("model"))
    extras = {k: body[k] for k in EXTRAS if body.get(k) is not None}

    # Log state size for debugging slow requests (DEBUG to avoid leaking sensitive data)
    state = body["state"]
    state_str = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    logger.debug("state: %d chars, preview: %r", len(state_str), state_str[:200])

    client = get_client()
    start = time.time()
    try:
        with _infer_lock:
            response = client.system_one(state=body["state"], questions=questions, model=model, **extras)
    except ValueError as e:  # unknown model, a Choice with one option, a Score with one level...
        raise RequestError(400, "api_usage_error", str(e)) from e
    latency_ms = (time.time() - start) * 1000
    logger.info("systemone: %d question(s), %.0f ms", len(questions), latency_ms)
    result = response.as_dict()
    # Log answer summary (DEBUG to avoid leaking sensitive data)
    answers_summary = {k: _answer_summary(v) for k, v in result.get("answers", {}).items()}
    logger.debug("answers: %s", answers_summary)
    result["jul"] = {**result.get("jul", {}), "latency_ms": round(latency_ms, 2)}
    return result


def _answer_summary(answer: dict) -> str:
    """Compact summary of an answer for logging."""
    t = answer.get("type", "?")
    if t == "noul":
        return f"noul={answer.get('noul', '?'):.2f}"
    if t == "choice":
        return f"choice={answer.get('choice', '?')} ({answer.get('confidence', 0):.0%})"
    if t == "score":
        return f"score={answer.get('score', '?'):.2f} ({answer.get('confidence', 0):.0%})"
    return str(answer)


def list_models() -> dict:
    """The Jev shape, which the official SDK validates: {"models": [{name, description, release_date}]}."""
    from jul.presets import PRESETS, fitted_presets
    fitted = {p.name for p in fitted_presets()}
    names = sorted(set(PRESETS) | fitted)
    today = time.strftime("%Y-%m-%d")
    models = [{"name": "jev-latest", "description": "Alias of the model this jul serve runs.",
               "release_date": today}]
    models += [{"name": n, "description": "JuL preset" + (" (added with jul models add)" if n in fitted else ""),
                "release_date": today} for n in names]
    return {"models": models}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # through the module logger, at DEBUG, not stderr
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _authorized(self) -> bool:
        """True when no key is configured, or the request carries it (Bearer or x-api-key)."""
        if _api_key is None:
            return True
        auth = self.headers.get("Authorization", "")
        bearer = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
        provided = bearer or self.headers.get("x-api-key", "")
        return hmac.compare_digest(provided.encode(), _api_key.encode())

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _refuse(self) -> None:
        if "Authorization" in self.headers or "x-api-key" in self.headers:
            self._send(401, {"error": {"type": "authentication_error",
                                       "message": "Cannot authenticate with the server: invalid API key"}})
        else:
            self._send(403, {"error": {"type": "authentication_error",
                                       "message": "Must supply an API key! (Authorization: Bearer <key>)"}})

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        if path not in ("/health", "/v1/health", "/v1/models"):
            self._send(404, {"error": {"type": "not_found", "message": f"no route {self.path}"}})
            return
        if not self._authorized():
            self._refuse()
            return
        if path == "/v1/models":
            self._send(200, list_models())
        else:
            self._send(200, {"status": "ok", "model": _client.model if _client else _default_model,
                             "ready": _client is not None})

    def do_POST(self) -> None:
        if self.path.split("?")[0].rstrip("/") not in SYSTEMONE_PATHS:
            self._send(404, {"error": {"type": "not_found", "message": f"no route {self.path}"}})
            return
        if not self._authorized():
            self._refuse()
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                raise RequestError(413, "max_tokens_exceeded", "request body too large")
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw or b"{}")
            except ValueError as e:
                raise RequestError(400, "api_usage_error", f"Invalid request. Invalid JSON: {e}") from e
            self._send(200, classify(body))
        except RequestError as e:
            self._send(e.status, e.payload())
        except Exception as e:  # noqa: BLE001 - any inference failure is a 500, with its type
            logger.exception("inference failed")
            self._send(500, {"error": {"type": type(e).__name__, "message": str(e)}})


def make_server(host: str = "127.0.0.1", port: int = 8577) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def serve(model: str | None = None, backend: str | None = None,
          host: str = "127.0.0.1", port: int = 8577, warmup: bool = True,
          api_key: str | None = None, escalate_to: str | None = None, escalate_model: str = "jev-latest",
          min_confidence: float = 0.8) -> None:
    """Run the server until interrupted. See the module docstring for the protocol.

    `escalate_to` (a System One server root) sends the answers below `min_confidence` there; its key
    is read from $JUL_ESCALATE_API_KEY, never from the command line.
    """
    global _default_model, _default_backend, _api_key, _escalate
    _default_model, _default_backend = model, backend
    _escalate = ({"url": escalate_to, "model": escalate_model, "min_confidence": min_confidence,
                  "api_key": os.environ.get("JUL_ESCALATE_API_KEY") or None} if escalate_to else None)
    _api_key = api_key or os.environ.get("JUL_API_KEY") or None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if host not in ("127.0.0.1", "::1", "localhost") and _api_key is None:
        logger.warning("jul serve is bound to %s WITHOUT an API key: anyone on the network can query "
                       "it. Set --api-key or $JUL_API_KEY.", host)
    if _api_key is not None:
        logger.info("API key authentication is enabled (Authorization: Bearer or x-api-key).")

    if warmup:
        get_client()

    server = make_server(host, port)
    logger.info("jul serve listening on http://%s:%d  (POST /v1/systemone, GET /v1/models, GET /health)",
                host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()
