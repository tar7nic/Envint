"""Block until every given URL answers with HTTP 200.  python -m scripts.wait_for URL [URL ...]"""
import sys
import time

import httpx


def wait(urls: list[str], timeout_s: float = 300, interval_s: float = 2) -> bool:
    deadline = time.monotonic() + timeout_s
    pending = list(urls)
    while pending and time.monotonic() < deadline:
        for u in list(pending):
            try:
                if httpx.get(u, timeout=3).status_code == 200:
                    pending.remove(u)
            except httpx.HTTPError:
                pass
        if pending:
            time.sleep(interval_s)
    return not pending


if __name__ == "__main__":
    ok = wait(sys.argv[1:])
    print("ready" if ok else "TIMEOUT waiting for services")
    sys.exit(0 if ok else 1)