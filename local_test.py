"""
One-command local test suite: python local_test.py

Exercises the real Flask routes (the same code Render serves) across every
data-availability mode. The failure modes are simulated by tripping the
Yahoo circuit breaker / stubbing out EDGAR, because a healthy home IP will
never hit them naturally.

  1. normal        - Yahoo healthy, full analysis (also warms the peer-median
                     and serve-stale caches for later tests)
  2. sec-fallback  - Yahoo quoteSummary "rate-limited" -> SEC EDGAR fundamentals,
                     including the computed extras (beta, TTM data, dividends)
  3. peer-cache    - SEC-fallback re-analysis of a ticker whose peer medians
                     were cached while Yahoo was healthy -> medians survive
  4. serve-stale   - Yahoo AND EDGAR dead -> labeled copy of the last analysis
  5. bad format    - "$$$" rejected with a 400 before touching any data source
  6. fake ticker   - "ZZZZZZ" refused with a clear "couldn't find" message
"""

import time

from app import app
import extraTrashTester as ett
import sec_fallback

client = app.test_client()
results = []


def check(name, passed, detail=""):
    results.append(passed)
    print(f"  {'PASS' if passed else 'FAIL'}  {name}  {detail}")


def force_yahoo_outage():
    ett._INFO_BREAKER["cooldown_until"] = time.time() + 99999


def heal_yahoo():
    ett._INFO_BREAKER["cooldown_until"] = 0


print("\n1. Normal analysis (Yahoo healthy) - MSFT ...")
r = client.get("/api/assessment?ticker=MSFT")
d = r.get_json()
check(
    "normal analysis",
    r.status_code == 200 and d.get("success") and not d.get("data_note"),
    f"{d.get('company_name')} | quality {d.get('overall_quality_score')} risk {d.get('generalized_risk_score')}",
)

print("\n2. SEC fallback (simulating Yahoo rate limit) - AAPL ...")
force_yahoo_outage()
ett._INFO_CACHE.pop("AAPL", None)
ett._TICKER_CACHE.pop("AAPL", None)
r = client.get("/api/assessment?ticker=AAPL")
d = r.get_json()
snap = d.get("metric_snapshot") or {}
check(
    "SEC fallback analysis",
    r.status_code == 200 and d.get("success") and "rate-limited" in (d.get("data_note") or ""),
    f"{d.get('company_name')} | quality {d.get('overall_quality_score')} | "
    f"sector {snap.get('Sector')} | revenue growth {snap.get('Revenue Growth')}",
)
check(
    "fallback computes beta",
    snap.get("Beta") not in (None, "N/A"),
    f"beta {snap.get('Beta')}",
)

print("\n3. Peer-median cache (SEC-fallback MSFT reuses medians from test 1) ...")
ett._INFO_CACHE.pop("MSFT", None)
ett._TICKER_CACHE.pop("MSFT", None)
r = client.get("/api/assessment?ticker=MSFT")
d = r.get_json()
peers = (d.get("peer_comparison") or {}).get("peers_used") or []
check(
    "peer medians survive throttle",
    r.status_code == 200 and d.get("success")
    and "rate-limited" in (d.get("data_note") or "") and len(peers) > 0,
    f"peers reused: {len(peers)}",
)

print("\n4. Serve-stale (Yahoo AND EDGAR dead) - MSFT again ...")
ett._INFO_CACHE.pop("MSFT", None)
ett._TICKER_CACHE.pop("MSFT", None)
_real_edgar = sec_fallback.build_info_from_edgar
sec_fallback.build_info_from_edgar = lambda *a, **k: {}
r = client.get("/api/assessment?ticker=MSFT")
d = r.get_json()
check(
    "serve-stale analysis",
    r.status_code == 200 and d.get("success") and d.get("stale") is True,
    f"stale={d.get('stale')} | quality {d.get('overall_quality_score')}",
)
sec_fallback.build_info_from_edgar = _real_edgar
heal_yahoo()

print("\n5. Malformed ticker ...")
r = client.get("/api/assessment?ticker=$$$")
check("bad-format 400", r.status_code == 400)

print("\n6. Fake ticker ...")
r = client.get("/api/assessment?ticker=ZZZZZZ")
d = r.get_json()
check(
    "fake-ticker refusal",
    not d.get("success") and "couldn't find" in d.get("error", ""),
)

print()
print("ALL PASS - safe to push." if all(results) else "SOMETHING FAILED - check above before pushing.")
