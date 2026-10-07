#!/usr/bin/env python3
"""Build coach-stats.json for learncedh.com/coaching from the TopDeck.gg API.

TopDeck has no player profile endpoint, so this searches EDH tournaments in
date windows and keeps every standings row that belongs to a coach. Results are
cached per event in coach-events.json, so nightly runs only re-fetch recent
weeks. Needs TOPDECK_API_KEY in the environment.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

API = "https://topdeck.gg/api/v2/tournaments"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COACHES_FILE = os.path.join(ROOT, "coaches", "coaches.json")
EVENTS_FILE = os.path.join(ROOT, "coach-events.json")
STATS_FILE = os.path.join(ROOT, "coach-stats.json")
CANDIDATES_FILE = os.path.join(ROOT, "coach-id-candidates.json")

BACKFILL_START = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp())
WINDOW = 14 * 86400      # days per search request
REFRESH = 28 * 86400     # always re-fetch this much recent history
PAUSE = 3                # seconds between requests (bulk search is rate limited)
COLUMNS = ["name", "id", "wins", "draws", "losses", "winsBracket", "lossesBracket", "byes"]


def load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")


def search(key, start, end):
    body = json.dumps({
        "game": "Magic: The Gathering", "format": "EDH",
        "start": start, "end": end, "columns": COLUMNS,
    }).encode()
    for attempt in range(6):
        req = urllib.request.Request(API, data=body, method="POST", headers={
            "Authorization": key, "Content-Type": "application/json",
            "User-Agent": "learncedh-coach-stats",
        })
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                wait = int(e.headers.get("Retry-After") or 30 * (attempt + 1))
                print(f"  HTTP {e.code}, waiting {wait}s")
                time.sleep(wait)
                continue
            print(f"  HTTP {e.code}: {e.read()[:300]!r}")
            raise
        except urllib.error.URLError as e:
            print(f"  network error {e}, retrying")
            time.sleep(15)
    raise RuntimeError("TopDeck search kept failing")


def norm(s):
    return " ".join((s or "").lower().split())


def main():
    key = os.environ.get("TOPDECK_API_KEY", "").strip()
    if not key:
        sys.exit("TOPDECK_API_KEY is not set")

    coaches = load(COACHES_FILE, {})["coaches"]
    ids = {c["topdeckId"]: c["name"] for c in coaches if c.get("topdeckId")}
    name_keys = {}
    for c in coaches:
        for n in [c["name"]] + c.get("aliases", []):
            name_keys[norm(n)] = c["name"]

    cache = load(EVENTS_FILE, {"fetchedThrough": BACKFILL_START, "events": {}})
    candidates = load(CANDIDATES_FILE, {})
    now = int(time.time())
    start = max(BACKFILL_START, cache["fetchedThrough"] - REFRESH)
    if os.environ.get("FULL_REBUILD") == "1":
        start, cache["events"] = BACKFILL_START, {}
    else:
        # A new coach id, or a coach whose id we don't know yet, needs their full
        # history (unless it was already saved while scanning by name).
        new_ids = set(ids) - set(cache.get("ids", []))
        unscanned = [c["name"] for c in coaches if not c.get("topdeckId") and c["name"] not in cache.get("scanned", [])]
        if any(i not in cache["events"] for i in new_ids) or unscanned:
            start = BACKFILL_START
    cache["ids"] = sorted(ids)
    cache["scanned"] = sorted(set(cache.get("scanned", [])) | {c["name"] for c in coaches if not c.get("topdeckId")})

    requests = 0
    while start < now:
        end = min(start + WINDOW, now)
        events = search(key, start, end)
        requests += 1
        print(f"{datetime.fromtimestamp(start, timezone.utc):%Y-%m-%d}: {len(events)} events")
        for ev in events:
            if ev.get("isLeague") or ev.get("isTeamEvent"):
                continue
            standings = ev.get("standings") or []
            top_cut = ev.get("topCut") or 0
            for place, row in enumerate(standings, 1):
                pid = row.get("id") or ""
                pname = norm(row.get("name"))
                matched = False
                for cname_key, cname in name_keys.items():
                    if cname_key and (cname_key == pname or cname_key in pname):
                        matched = True
                        cand = candidates.setdefault(cname, {}).setdefault(pid or "(no id)", {"names": [], "events": []})
                        if row.get("name") not in cand["names"]:
                            cand["names"].append(row.get("name"))
                        if ev["TID"] not in cand["events"]:
                            cand["events"].append(ev["TID"])
                # Also keep results for name matches, so a coach's history is
                # already saved when their id is added later.
                if pid not in ids and not (matched and pid):
                    continue
                bracket_games = (row.get("winsBracket") or 0) + (row.get("lossesBracket") or 0)
                cache["events"].setdefault(pid, {})[ev["TID"]] = {
                    "name": ev.get("tournamentName"),
                    "date": ev.get("startDate"),
                    "players": len(standings),
                    "place": place,
                    "topCut": top_cut,
                    "madeTopCut": bool(bracket_games > 0 or (top_cut and place <= top_cut)),
                    "wins": row.get("wins") or 0,
                    "losses": row.get("losses") or 0,
                    "draws": row.get("draws") or 0,
                }
        cache["fetchedThrough"] = end
        start = end
        time.sleep(PAUSE)

    out = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source": "https://topdeck.gg", "coaches": {}}
    for c in coaches:
        evs = list(cache["events"].get(c.get("topdeckId") or "", {}).values())
        if not evs:
            continue
        w = sum(e["wins"] for e in evs)
        l = sum(e["losses"] for e in evs)
        d = sum(e["draws"] for e in evs)
        # Match TopDeck's profile numbers: a win is 1st place in an event with a
        # top cut, and conversion only counts events that had a top cut.
        cuts = sum(1 for e in evs if e["madeTopCut"])
        cut_events = sum(1 for e in evs if e["topCut"] or e["madeTopCut"])
        wins = [e for e in evs if e["place"] == 1 and e["madeTopCut"]]
        best = max(wins, key=lambda e: e["players"]) if wins else None
        out["coaches"][c["name"]] = {
            "topdeckId": c["topdeckId"],
            "profile": c.get("profile"),
            "events": len(evs),
            "eventWins": len(wins),
            "topCuts": cuts,
            "conversion": round(cuts / cut_events, 4) if cut_events else 0,
            "games": {"wins": w, "losses": l, "draws": d},
            "gameWinRate": round(w / (w + l + d), 4) if w + l + d else 0,
            "biggestWin": {"name": best["name"].strip(), "players": best["players"]} if best else None,
        }

    save(EVENTS_FILE, cache)
    save(STATS_FILE, out)
    save(CANDIDATES_FILE, candidates)
    print(f"{requests} requests. " + json.dumps(out["coaches"], indent=1))


if __name__ == "__main__":
    main()
