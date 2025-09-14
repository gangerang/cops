#!/usr/bin/env python3
"""
Waze alerts extractor using Z/X/Y tiles with persistent "settled tiles" and on-the-fly refinement.

Behavior
--------
- Uses Web Mercator slippy tiles (Z/X/Y) to define query bboxes.
- If a saved settled tile list exists, uses it; otherwise, runs a refinement pass to build one.
- On every run, queries all settled tiles to extract alerts.
  * If any settled tile now returns >= threshold alerts, it is split (if z < max_zoom)
    and its children are queried recursively until all children are under threshold
    (or max_zoom is reached). The refined children replace the parent in the new
    settled tile list for next runs.
- Deduplicates alerts by `id` and saves them to a timestamped JSON file.
- Saves the updated settled tiles to disk for future runs.
- Prints per-tile status during querying.

Requirements
------------
pip install requests

Example
-------
python waze_zxy_extract.py \
  --left 150.50 --right 151.50 --bottom -34.20 --top -33.40 \
  --base-zoom 12 --max-zoom 17 --threshold 150 --env row \
  --state-path ./settled_tiles.json --out-dir ./out --overlap-deg 0.002
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import requests
from datetime import datetime, timezone


WAZE_URL = "https://www.waze.com/live-map/api/georss"
HTTP_TIMEOUT = 12  # seconds
DEFAULT_HEADERS = {"User-Agent": "waze-zxy-extractor/1.0"}


# ------------------------------ Z/X/Y math ------------------------------

@dataclass(frozen=True, order=True)
class Tile:
    z: int
    x: int
    y: int

    def children(self) -> Tuple["Tile", "Tile", "Tile", "Tile"]:
        z = self.z + 1
        X = self.x * 2
        Y = self.y * 2
        return (
            Tile(z, X,     Y    ),  # NW
            Tile(z, X + 1, Y    ),  # NE
            Tile(z, X,     Y + 1),  # SW
            Tile(z, X + 1, Y + 1),  # SE
        )


def tile_bounds_latlon(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """
    Returns (left, right, bottom, top) in lon/lat for a slippy tile (Z/X/Y).
    """
    n = 2 ** z

    def lon(x_: int) -> float:
        return x_ / n * 360.0 - 180.0

    def lat(y_: int) -> float:
        t = math.pi * (1 - 2 * y_ / n)
        return math.degrees(math.atan(math.sinh(t)))

    left = lon(x)
    right = lon(x + 1)
    top = lat(y)
    bottom = lat(y + 1)
    return left, right, bottom, top


def lonlat_to_tile_xy(z: int, lon: float, lat: float) -> Tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tiles_covering_bbox(z: int, left: float, right: float, bottom: float, top: float) -> List[Tile]:
    """
    All tiles at zoom z intersecting the bbox (lon/lat).
    """
    x0, y0 = lonlat_to_tile_xy(z, left, top)      # NW
    x1, y1 = lonlat_to_tile_xy(z, right, bottom)  # SE
    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)

    tiles: List[Tile] = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append(Tile(z, x, y))
    return tiles


def tile_intersects_bbox(tile: Tile, left: float, right: float, bottom: float, top: float) -> bool:
    l, r, b, t = tile_bounds_latlon(tile.z, tile.x, tile.y)
    return (r >= left and l <= right and t >= bottom and b <= top)


# ------------------------------ Waze API ------------------------------

def query_waze_alerts(left: float, right: float, bottom: float, top: float, env: str) -> List[dict]:
    """
    Query Waze alerts for a bbox; returns the list of alert objects.
    """
    params = {
        "types": "alerts",
        "env": env,
        "left": f"{left:.10f}",
        "right": f"{right:.10f}",
        "bottom": f"{bottom:.10f}",
        "top": f"{top:.10f}",
    }
    r = requests.get(WAZE_URL, params=params, headers=DEFAULT_HEADERS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    try:
        data = r.json()
    except requests.exceptions.JSONDecodeError:
        data = json.loads(r.text)
    alerts = data.get("alerts", []) if isinstance(data, dict) else []
    return alerts


# ------------------------------ Persistence ------------------------------

def load_settled_tiles(state_path: Path) -> Set[Tile]:
    if not state_path.exists():
        return set()
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    tiles = {Tile(**t) for t in raw.get("tiles", [])}
    return tiles


def save_settled_tiles(state_path: Path, tiles: Set[Tile]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tiles": [asdict(t) for t in sorted(tiles)]}
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_alerts(out_dir: Path, alerts_by_id: Dict[str, dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"waze_alerts_{ts}.json"
    # Save as a flat list of alerts
    data = {"generated_at": ts, "count": len(alerts_by_id), "alerts": list(alerts_by_id.values())}
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ------------------------------ Refinement helpers ------------------------------

def clipped_bounds_with_overlap(tile: Tile, bbox: Tuple[float, float, float, float], overlap_deg: float) -> Tuple[float, float, float, float]:
    left, right, bottom, top = bbox
    l, r, b, t = tile_bounds_latlon(tile.z, tile.x, tile.y)
    # clip to target bbox to avoid querying outside
    l = max(l, left)
    r = min(r, right)
    b = max(b, bottom)
    t = min(t, top)
    # pad slightly to avoid edge misses
    return (l - overlap_deg, r + overlap_deg, b - overlap_deg, t + overlap_deg)


def refine_to_settled(
    bbox: Tuple[float, float, float, float],
    base_zoom: int,
    threshold: int,
    overlap_deg: float,
    env: str,
    max_zoom: int,
) -> Set[Tile]:
    """
    Build a settled tile set from scratch by splitting tiles that exceed threshold
    until all tiles are under threshold or max_zoom is reached.
    """
    left, right, bottom, top = bbox
    frontier: Set[Tile] = set(tiles_covering_bbox(base_zoom, left, right, bottom, top))
    settled: Set[Tile] = set()

    cycle = 1
    while frontier:
        print(f"\n[Build] Cycle {cycle} | Querying {len(frontier)} tile(s)")
        next_frontier: Set[Tile] = set()

        for tile in sorted(frontier):
            l, r, b, t = clipped_bounds_with_overlap(tile, bbox, overlap_deg)
            try:
                alerts = query_waze_alerts(l, r, b, t, env)
                count = len(alerts)
            except Exception as e:
                print(f"  z{tile.z}/{tile.x}/{tile.y} -> ERROR during build: {e}")
                count = 0

            if count >= threshold and tile.z < max_zoom:
                print(f"  z{tile.z}/{tile.x}/{tile.y} -> {count} | split for build")
                for child in tile.children():
                    if tile_intersects_bbox(child, left, right, bottom, top):
                        next_frontier.add(child)
            else:
                print(f"  z{tile.z}/{tile.x}/{tile.y} -> {count} | settled for build")
                settled.add(tile)

        frontier = next_frontier
        cycle += 1

    print(f"\n[Build] Settled tile count: {len(settled)}")
    return settled


# ------------------------------ Extraction run ------------------------------

def extract_with_refinement(
    bbox: Tuple[float, float, float, float],
    settled_tiles: Set[Tile],
    threshold: int,
    overlap_deg: float,
    env: str,
    max_zoom: int,
) -> Tuple[Set[Tile], Dict[str, dict]]:
    """
    Query all settled tiles. If any returns >= threshold, split (if z < max_zoom) and
    query children recursively until all children are under threshold. Returns the new
    settled tile set and a dict of alerts by id (deduped).
    """
    left, right, bottom, top = bbox
    alerts_by_id: Dict[str, dict] = {}
    new_settled: Set[Tile] = set()

    # Work queue starts with the currently settled tiles
    stack: List[Tile] = list(sorted(settled_tiles))

    print(f"\n[Run] Starting extraction on {len(stack)} settled tile(s)")
    while stack:
        tile = stack.pop()
        l, r, b, t = clipped_bounds_with_overlap(tile, bbox, overlap_deg)

        try:
            alerts = query_waze_alerts(l, r, b, t, env)
            count = len(alerts)
        except Exception as e:
            print(f"  z{tile.z}/{tile.x}/{tile.y} -> ERROR during run: {e}")
            alerts = []
            count = 0

        # Deduplicate by 'id'
        for a in alerts:
            a_id = a.get("id")
            if not a_id:
                # Fallback: build a poor-man's id hash if missing
                # (type + coords + pubMillis) to reduce dup risk
                loc = a.get("location") or {}
                a_id = f"{a.get('type','ALERT')}_{loc.get('x')}_{loc.get('y')}_{a.get('pubMillis')}"
            alerts_by_id[a_id] = a

        if count >= threshold and tile.z < max_zoom:
            print(f"  z{tile.z}/{tile.x}/{tile.y} -> {count} alerts | split and re-query children")
            for child in tile.children():
                if tile_intersects_bbox(child, left, right, bottom, top):
                    stack.append(child)
        else:
            status = "settled" if count < threshold else f"cap at z{tile.z} (max_zoom reached)"
            print(f"  z{tile.z}/{tile.x}/{tile.y} -> {count} alerts | {status}")
            new_settled.add(tile)

    print(f"[Run] New settled tile count: {len(new_settled)} | deduped alerts: {len(alerts_by_id)}")
    return new_settled, alerts_by_id


# ------------------------------ CLI & Main ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Waze Z/X/Y extractor with persistent settled tiles and on-the-fly refinement")
    p.add_argument("--left", type=float, default=150.50, help="bbox left (lon)")
    p.add_argument("--right", type=float, default=151.50, help="bbox right (lon)")
    p.add_argument("--bottom", type=float, default=-34.20, help="bbox bottom (lat)")
    p.add_argument("--top", type=float, default=-33.40, help="bbox top (lat)")
    p.add_argument("--base-zoom", type=int, default=12, help="base zoom for initial build if no state exists")
    p.add_argument("--max-zoom", type=int, default=17, help="max zoom to split tiles to during build/run")
    p.add_argument("--threshold", type=int, default=150, help="alerts >= threshold triggers split/refine")
    p.add_argument("--overlap-deg", type=float, default=0.002, help="small overlap (deg) to avoid edge misses")
    p.add_argument("--env", type=str, default="row", help="Waze env (row, usa, il, etc.)")
    p.add_argument("--state-path", type=Path, default=Path("./settled_tiles.json"), help="path to save/load settled tiles")
    p.add_argument("--out-dir", type=Path, default=Path("./out"), help="directory to write timestamped alerts JSON")
    return p.parse_args()


def main():
    args = parse_args()
    bbox = (args.left, args.right, args.bottom, args.top)

    # 1) Load or build settled tiles
    settled_tiles = load_settled_tiles(args.state_path)
    if settled_tiles:
        print(f"[Init] Loaded {len(settled_tiles)} settled tile(s) from {args.state_path}")
    else:
        print("[Init] No settled tiles found. Building a settled grid...")
        settled_tiles = refine_to_settled(
            bbox=bbox,
            base_zoom=args.base_zoom,
            threshold=args.threshold,
            overlap_deg=args.overlap_deg,
            env=args.env,
            max_zoom=args.max_zoom,
        )
        save_settled_tiles(args.state_path, settled_tiles)
        print(f"[Init] Saved initial settled tiles to {args.state_path}")

    # 2) Run extraction with on-the-fly refinement of any tiles now over threshold
    new_settled, alerts_by_id = extract_with_refinement(
        bbox=bbox,
        settled_tiles=settled_tiles,
        threshold=args.threshold,
        overlap_deg=args.overlap_deg,
        env=args.env,
        max_zoom=args.max_zoom,
    )

    # 3) Persist updated settled tiles for next run
    save_settled_tiles(args.state_path, new_settled)
    print(f"[Save] Updated settled tiles saved to {args.state_path}")

    # 4) Save deduplicated alerts with timestamped filename
    out_path = save_alerts(args.out_dir, alerts_by_id)
    print(f"[Save] Alerts saved to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
