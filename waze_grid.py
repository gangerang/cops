#!/usr/bin/env python3
"""
Improved Waze query algorithm using Z/X/Y tiles and adaptive refinement

What it does
------------
- Uses standard Web Mercator slippy tiles (Z/X/Y) to define query bounding boxes.
- Starts from a base zoom and covers a target bbox (Sydney by default).
- Runs cycles. In each cycle:
  * Only tiles that were ABOVE the threshold in the previous cycle are queried.
  * Tiles under the threshold are considered "settled" and not re-queried.
  * Tiles at/above the threshold are split into their 4 children (Z+1) for the next cycle.
- Prints per-tile status (count and whether it will be broken down).
- Stops automatically when all tiles are under the threshold (i.e., no frontier tiles remain),
  or when --max-cycles is reached (safety cap).

Notes
-----
- This script focuses on the querying & refinement algorithm. It does not persist alerts.
- You can persist the final settled tiles if you want to reuse the grid later (see TODO).
- The Waze endpoint used: https://www.waze.com/live-map/api/georss?types=alerts&env=row&left=..&right=..&bottom=..&top=..

Requirements
------------
pip install requests

Usage
-----
python zxy_waze_refine.py \
  --threshold 150 \
  --base-zoom 12 \
  --max-cycles 6

Optional:
  --left 150.50 --right 151.50 --bottom -34.20 --top -33.40  # custom bbox
  --overlap-deg 0.002                                        # ~200m edge buffer
  --env row                                                  # or usa, il, etc (Waze env)
"""

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import requests


WAZE_URL = "https://www.waze.com/live-map/api/georss"
HTTP_TIMEOUT = 12  # seconds


# ------------------------------ Tile math ------------------------------

@dataclass(frozen=True)
class Tile:
    z: int
    x: int
    y: int

    def children(self) -> Tuple["Tile", "Tile", "Tile", "Tile"]:
        """
        Return the 4 children of this tile at zoom z+1:
        NW, NE, SW, SE (conventional layout for readability).
        """
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
    Returns (left, right, bottom, top) in lon/lat for a Web Mercator slippy tile.
    """
    n = 2 ** z

    def lon(x_: int) -> float:
        return x_ / n * 360.0 - 180.0

    def lat(y_: int) -> float:
        # inverse Web Mercator
        t = math.pi * (1 - 2 * y_ / n)
        return math.degrees(math.atan(math.sinh(t)))

    left = lon(x)
    right = lon(x + 1)
    top = lat(y)
    bottom = lat(y + 1)
    return left, right, bottom, top


def lonlat_to_tile_xy(z: int, lon: float, lat: float) -> Tuple[int, int]:
    """
    Convert lon/lat to slippy tile x,y at zoom z.
    """
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tiles_covering_bbox(z: int, left: float, right: float, bottom: float, top: float) -> List[Tile]:
    """
    Return all tiles at zoom z that intersect the bbox (lon/lat).
    """
    x0, y0 = lonlat_to_tile_xy(z, left, top)      # northwest corner
    x1, y1 = lonlat_to_tile_xy(z, right, bottom)  # southeast corner
    # Ensure proper ordering
    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)

    tiles = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append(Tile(z, x, y))
    return tiles


# ------------------------------ Waze API ------------------------------

def query_waze_alerts(left: float, right: float, bottom: float, top: float, env: str) -> int:
    """
    Query Waze alerts for a bbox; return the number of alerts (int).
    We only care about counts for refinement decisions here.
    """
    params = {
        "types": "alerts",
        "env": env,
        "left": f"{left:.10f}",
        "right": f"{right:.10f}",
        "bottom": f"{bottom:.10f}",
        "top": f"{top:.10f}",
    }
    r = requests.get(WAZE_URL, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    try:
        data = r.json()
    except requests.exceptions.JSONDecodeError:
        # occasionally text; try manual
        import json
        data = json.loads(r.text)
    alerts = data.get("alerts", [])
    return len(alerts)


# ------------------------------ Core algorithm ------------------------------

def run_cycles(
    sydney_bbox: Tuple[float, float, float, float],
    base_zoom: int,
    threshold: int,
    overlap_deg: float,
    env: str,
    max_cycles: int,
) -> Tuple[Set[Tile], Dict[Tile, int]]:
    """
    Run cycles of queries/refinement until all tiles are under threshold
    or we hit max_cycles.

    Returns:
      settled_tiles: set of all tiles that ended under threshold
      counts: dict of last known counts per tile (includes settled & last frontier queried)
    """
    left, right, bottom, top = sydney_bbox

    # Initial frontier: all tiles at base zoom that intersect the bbox
    frontier: Set[Tile] = set(tiles_covering_bbox(base_zoom, left, right, bottom, top))
    settled: Set[Tile] = set()
    counts: Dict[Tile, int] = {}

    for cycle in range(1, max_cycles + 1):
        if not frontier:
            print(f"\nAll tiles are under threshold. Stopping at start of cycle {cycle}.")
            break

        print(f"\n=== Cycle {cycle} | Querying {len(frontier)} tile(s) ===")

        next_frontier: Set[Tile] = set()

        for tile in sorted(frontier, key=lambda t: (t.z, t.x, t.y)):
            l, r, b, ttop = tile_bounds_latlon(tile.z, tile.x, tile.y)

            # Clip to target bbox to avoid querying outside your area (optional, keeps load down)
            l = max(l, left)
            r = min(r, right)
            b = max(b, bottom)
            ttop = min(ttop, top)

            # Apply small overlap to avoid missing edge alerts
            l -= overlap_deg
            r += overlap_deg
            b -= overlap_deg
            ttop += overlap_deg

            try:
                count = query_waze_alerts(l, r, b, ttop, env)
            except Exception as e:
                print(f"  z{tile.z}/{tile.x}/{tile.y} -> ERROR: {e}")
                count = 0  # treat failures as zero to avoid infinite loops; you may want retries

            counts[tile] = count

            if count >= threshold:
                print(f"  z{tile.z}/{tile.x}/{tile.y} -> {count} alerts  | split -> 4 children next cycle")
                # Only split if we can meaningfully refine: create 4 children
                for child in tile.children():
                    # Only keep children that intersect our overall bbox
                    cl, cr, cb, ct = tile_bounds_latlon(child.z, child.x, child.y)
                    if (cr >= left and cl <= right and ct >= bottom and cb <= top):
                        next_frontier.add(child)
            else:
                print(f"  z{tile.z}/{tile.x}/{tile.y} -> {count} alerts  | settled (no further queries)")
                settled.add(tile)

        frontier = next_frontier  # only tiles that were above threshold get re-queried
        # Loop continues until frontier is empty or max_cycles reached

    if frontier:
        print(f"\nReached max cycles ({max_cycles}) with {len(frontier)} frontier tile(s) still above threshold.")
        # Optionally, you could add them to settled anyway, but typically you keep them separate
    else:
        print("\nRefinement complete. All tiles under threshold.")

    return settled, counts


# ------------------------------ CLI ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Waze Z/X/Y tile-based adaptive query/refinement")
    p.add_argument("--left", type=float, default=150.50, help="bbox left (lon)")
    p.add_argument("--right", type=float, default=151.50, help="bbox right (lon)")
    p.add_argument("--bottom", type=float, default=-34.20, help="bbox bottom (lat)")
    p.add_argument("--top", type=float, default=-33.40, help="bbox top (lat)")
    p.add_argument("--base-zoom", type=int, default=12, help="base zoom level for initial tiles")
    p.add_argument("--threshold", type=int, default=150, help="split threshold (alerts >= threshold => split)")
    p.add_argument("--overlap-deg", type=float, default=0.002, help="small overlap (degrees) to avoid edge misses")
    p.add_argument("--env", type=str, default="row", help="Waze env (row, usa, il, etc.)")
    p.add_argument("--max-cycles", type=int, default=6, help="safety cap on number of cycles")
    return p.parse_args()


def main():
    args = parse_args()
    bbox = (args.left, args.right, args.bottom, args.top)

    settled, counts = run_cycles(
        sydney_bbox=bbox,
        base_zoom=args.base_zoom,
        threshold=args.threshold,
        overlap_deg=args.overlap_deg,
        env=args.env,
        max_cycles=args.max_cycles,
    )

    # Summary
    print("\n--- Summary ---")
    print(f"Settled tiles: {len(settled)}")
    if settled:
        # Show a few examples
        examples = list(sorted(settled, key=lambda t: (t.z, t.x, t.y)))
        for t in examples:
            c = counts.get(t, -1)
            print(f"  z{t.z}/{t.x}/{t.y} -> {c} alerts")

    # TODO (optional): persist the settled grid for future reuse (e.g., JSON file with z/x/y and last counts).
    # You can later reload and ONLY query those tiles you care about (e.g., periodically sample),
    # or use the settled set as your "operational grid" until conditions change.


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
