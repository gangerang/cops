#!/usr/bin/env python3
"""
Convert a saved Waze alerts JSON (from the extractor) into a POLICE-only GeoJSON.

- Filters alerts where type == "POLICE"
- Geometry: Point from alert.location {x: lon, y: lat}
- Flattens properties and removes the 'comments' list
- Adds:
    * pub_ts: ISO8601 UTC string converted from pubMillis (if present)
    * comment_first_ts: ISO8601 UTC string from min(reportMillis) in comments (if any)
    * comment_last_ts:  ISO8601 UTC string from max(reportMillis) in comments (if any)

Usage:
    python json_to_geojson.py input.json output.geojson
"""

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def epoch_millis_to_iso(millis: Optional[int]) -> Optional[str]:
    if millis is None:
        return None
    try:
        # Some fields might come in as strings; coerce to int
        ms = int(millis)
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def comments_min_max_ts(comments: Any) -> (Optional[str], Optional[str]):
    if not isinstance(comments, list) or not comments:
        return None, None
    times = []
    for c in comments:
        if isinstance(c, dict) and "reportMillis" in c:
            try:
                times.append(int(c["reportMillis"]))
            except Exception:
                continue
    if not times:
        return None, None
    return epoch_millis_to_iso(min(times)), epoch_millis_to_iso(max(times))


def alert_to_feature(alert: Dict) -> Optional[Dict]:
    # Filter by type == POLICE
    if alert.get("type") != "POLICE":
        return None

    loc = alert.get("location") or {}
    lon = loc.get("x")
    lat = loc.get("y")
    if lon is None or lat is None:
        return None  # skip if no geometry

    # Compute timestamps
    pub_ts = epoch_millis_to_iso(alert.get("pubMillis"))
    c_first, c_last = comments_min_max_ts(alert.get("comments"))

    # Prepare properties: copy scalar fields except 'location' and 'comments'
    props = {}
    for k, v in alert.items():
        if k in ("location", "comments"):
            continue
        # keep original pubMillis, but add pub_ts separately
        props[k] = v

    props["pub_ts"] = pub_ts
    props["comment_first_ts"] = c_first
    props["comment_last_ts"] = c_last

    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }
    return feature


def convert(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    alerts = data.get("alerts", [])
    features: List[Dict] = []

    for a in alerts:
        if not isinstance(a, dict):
            continue
        feat = alert_to_feature(a)
        if feat:
            features.append(feat)

    fc = {"type": "FeatureCollection", "features": features}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(features)} POLICE features → {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Convert Waze alerts JSON to POLICE-only GeoJSON")
    ap.add_argument("input", help="Input JSON file (from extractor)")
    ap.add_argument("output", help="Output GeoJSON file")
    args = ap.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
