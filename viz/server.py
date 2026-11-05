#!/usr/bin/env python3
"""Tiny stdlib HTTP server for the UltiSim replay viewer.

    python viz/server.py [--port 8765] [--runs PATH]

Routes
    /                 -> files in viz/ (index.html, viewer.js, style.css)
    /runs/...         -> files in the project's runs/ directory (Cache-Control: no-store)
    /api/replays      -> JSON list [{path, size, mtime}] of every runs/**/*.json that
                         looks like an ultisim replay, newest first
    /api/runs         -> JSON list of evolution runs (viz/RUN_FORMAT.md): every directory
                         directly under runs/ holding meta.json or history.jsonl, as
                         {name, mtime, state, gen, planned, kind, has_replay}, newest first
    /dashboard.html   -> live dashboard for an evolution run (?run=<name>)

Files under /runs/ are read fully into memory and closed before being sent, so a writer
doing temp-file + os.replace() on Windows is blocked for as short a time as possible.

Binds to 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import email.utils
import io
import json
import os
import posixpath
import re
import stat
import sys
import threading
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(VIZ_DIR)
DEFAULT_RUNS_DIR = os.path.join(ROOT_DIR, "runs")

# The writer puts "format" first, so the first 200 bytes are normally enough. We sniff a
# little more (still cheap) in case a writer emits other small keys before it.
SNIFF_BYTES = 4096
_FORMAT_RE = re.compile(rb'"format"\s*:\s*"ultisim-replay"')

_sniff_cache: dict[str, tuple[int, float, bool]] = {}
_sniff_lock = threading.Lock()


def looks_like_replay(path: str, size: int, mtime: float) -> bool:
    """Cheap check (cached by size+mtime): does the file start like an ultisim replay?"""
    with _sniff_lock:
        hit = _sniff_cache.get(path)
    if hit is not None and hit[0] == size and hit[1] == mtime:
        return hit[2]
    try:
        with open(path, "rb") as f:
            head = f.read(SNIFF_BYTES)
    except OSError:
        return False
    ok = bool(_FORMAT_RE.search(head))
    with _sniff_lock:
        _sniff_cache[path] = (size, mtime, ok)
    return ok


def list_replays(runs_dir: str) -> list[dict]:
    out: list[dict] = []
    if not os.path.isdir(runs_dir):
        return out
    for dirpath, dirnames, filenames in os.walk(runs_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if not fn.lower().endswith(".json"):
                continue
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            if st.st_size == 0 or not looks_like_replay(full, st.st_size, st.st_mtime):
                continue
            rel = os.path.relpath(full, runs_dir).replace(os.sep, "/")
            out.append({
                "path": "/runs/" + urllib.parse.quote(rel),
                "size": st.st_size,
                "mtime": st.st_mtime,
            })
    out.sort(key=lambda e: e["mtime"], reverse=True)
    return out


RUN_MARKERS = ("meta.json", "history.jsonl")
TAIL_BYTES = 65536


def _read_json(path: str):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return json.loads(data.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _last_history_gen(path: str):
    """gen of the last *complete* line of history.jsonl (a trailing partial line is ignored)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - TAIL_BYTES))
            tail = f.read()
    except OSError:
        return None
    lines = tail.split(b"\n")
    lines.pop()   # text after the final newline is either empty or a line still being written
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue   # first line of the tail window may be cut; keep looking
        if isinstance(obj, dict) and isinstance(obj.get("gen"), int):
            return obj["gen"]
    return None


def list_runs(runs_dir: str) -> list[dict]:
    out: list[dict] = []
    try:
        names = os.listdir(runs_dir)
    except OSError:
        return out
    for name in names:
        if name.startswith("."):
            continue
        d = os.path.join(runs_dir, name)
        if not os.path.isdir(d):
            continue
        try:
            files = os.listdir(d)
        except OSError:
            continue
        if not any(m in files for m in RUN_MARKERS):
            continue
        mtime = 0.0
        for fn in files:
            try:
                st = os.stat(os.path.join(d, fn))
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                mtime = max(mtime, st.st_mtime)
        status = _read_json(os.path.join(d, "status.json")) if "status.json" in files else None
        meta = _read_json(os.path.join(d, "meta.json")) if "meta.json" in files else None
        status = status if isinstance(status, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        gen = status.get("gen")
        if not isinstance(gen, int) and "history.jsonl" in files:
            gen = _last_history_gen(os.path.join(d, "history.jsonl"))
        out.append({
            "name": name,
            "mtime": mtime,
            "state": status.get("state"),
            "gen": gen if isinstance(gen, int) else None,
            "planned": status.get("generations_planned", meta.get("generations_planned")),
            "kind": meta.get("kind"),
            "has_replay": "latest_best.json" in files,
        })
    out.sort(key=lambda e: e["mtime"], reverse=True)
    return out


def _safe_join(base: str, rel: str) -> str:
    """Join a URL path onto base, dropping '.', '..', drive letters and backslash tricks."""
    parts = []
    for word in rel.split("/"):
        if not word or word in (os.curdir, os.pardir):
            continue
        if os.path.dirname(word) or ":" in word or "\\" in word:
            continue
        parts.append(word)
    return os.path.join(base, *parts)


def make_handler(runs_dir: str):
    class Handler(SimpleHTTPRequestHandler):
        extensions_map = {
            **SimpleHTTPRequestHandler.extensions_map,
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".mjs": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".jsonl": "text/plain; charset=utf-8",
            ".md": "text/plain; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
        }

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=VIZ_DIR, **kwargs)

        # ---- routing -------------------------------------------------------------
        def _url_path(self) -> str:
            return urllib.parse.urlsplit(self.path).path

        def _is_runs(self) -> bool:
            p = self._url_path()
            return p == "/runs" or p.startswith("/runs/")

        def translate_path(self, path: str) -> str:
            p = urllib.parse.unquote(urllib.parse.urlsplit(path).path)
            trailing = p.endswith("/")
            p = posixpath.normpath(p)
            if p == "/runs" or p.startswith("/runs/"):
                out = _safe_join(runs_dir, p[len("/runs"):])
            else:
                out = _safe_join(VIZ_DIR, p)
            if trailing:
                out += os.sep
            return out

        def _api(self):
            p = self._url_path().rstrip("/")
            if p == "/api/replays":
                return list_replays(runs_dir)
            if p == "/api/runs":
                return list_runs(runs_dir)
            return None

        def do_GET(self):
            obj = self._api()
            if obj is not None:
                return self._send_json(obj)
            return super().do_GET()

        def do_HEAD(self):
            obj = self._api()
            if obj is not None:
                return self._send_json(obj, head_only=True)
            return super().do_HEAD()

        def send_head(self):
            # Run files: read the whole file and close it before replying, so the evolution
            # process can os.replace() it (Windows refuses while we hold it open).
            if not self._is_runs():
                return super().send_head()
            path = self.translate_path(self.path)
            if os.path.isdir(path) or not os.path.isfile(path):
                return super().send_head()
            try:
                with open(path, "rb") as f:
                    st = os.fstat(f.fileno())
                    data = f.read()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return None
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Last-Modified", email.utils.formatdate(st.st_mtime, usegmt=True))
            self.end_headers()
            return io.BytesIO(data)

        def _send_json(self, obj, head_only: bool = False):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head_only:
                self.wfile.write(body)

        def end_headers(self):
            p = self._url_path()
            if self._is_runs() or p.startswith("/api/"):
                self.send_header("Cache-Control", "no-store")
            else:
                # viewer files: always revalidate so edits show up on reload
                self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        # ---- quieter logging ----------------------------------------------------------
        def log_request(self, code="-", size="-"):
            # Live mode polls every few seconds; don't spam the console with those.
            if isinstance(code, HTTPStatus):
                code = code.value
            if str(code) in ("200", "304") and (
                self._url_path().startswith("/api/") or "_=" in urllib.parse.urlsplit(self.path).query
                or self.command == "HEAD"
            ):
                return
            super().log_request(code, size)

        def handle(self):
            try:
                super().handle()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets two servers silently share a port; don't.
    allow_reuse_address = os.name != "nt"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Serve the UltiSim replay viewer.")
    ap.add_argument("--port", type=int, default=8765, help="port to listen on (default 8765)")
    ap.add_argument("--runs", default=DEFAULT_RUNS_DIR, help="runs directory to serve at /runs/")
    args = ap.parse_args(argv)

    runs_dir = os.path.abspath(args.runs)
    try:
        httpd = Server(("127.0.0.1", args.port), make_handler(runs_dir))
    except OSError as e:
        print(f"Could not bind 127.0.0.1:{args.port}: {e}", file=sys.stderr)
        return 1
    print(f"UltiSim viewer: http://127.0.0.1:{args.port}/   (runs: {runs_dir})", flush=True)
    if not os.path.isdir(runs_dir):
        print("  note: runs directory does not exist yet; try `python viz/make_sample.py`", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
