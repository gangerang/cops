#!/usr/bin/env python3
"""
Hybrid (base-grid + adaptive refinement) Waze alert poller for Sydney.
- Queries the Waze Live Map GeoRSS API by bounding boxes (BBOX).
- If a tile returns >= SPLIT_THRESHOLD alerts, it will be split into 4 children next run.
- Saves the current tile grid to grid_state.json so it can be reused in future runs.
- Outputs GeoJSON files for the polled grid and alerts, and writes a Leaflet map HTML showing both.
 
Usage:
    python hybrid_waze_grid.py --once
    python hybrid_waze_grid.py --cycles 1 --sleep-sec 180
 
Notes:
- Requires: requests
- By default, runs a single cycle over Greater Sydney and writes outputs to ./out/
- You can change the bbox, thresholds, and min tile size below.
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import requests

# ----------------------------- Configuration -----------------------------

# Greater Sydney rough bbox (lon/lat)
SYDNEY_LEFT  = 140
SYDNEY_RIGHT = 154
SYDNEY_BOTTOM = -37.5
SYDNEY_TOP    = -29
# SYDNEY_LEFT  = 150.50
# SYDNEY_RIGHT = 151.50
# SYDNEY_BOTTOM = -34.20
# SYDNEY_TOP    = -33.40

# Base grid
BASE_ROWS = 1
BASE_COLS = 1

# API settings
WAZE_URL = "https://www.waze.com/live-map/api/georss"
WAZE_PARAMS_TEMPLATE = {
    "env": "row",
    "types": "alerts"
}

# Hybrid logic tuning
SPLIT_THRESHOLD = 150       # split when count >= this
MERGE_THRESHOLD = 50        # (optional) merge if all 4 kids are quiet; basic support provided
MIN_TILE_WIDTH_DEG  = 0.02  # ~2km lon at Sydney lat
MIN_TILE_HEIGHT_DEG = 0.02  # ~2km lat
OVERLAP_DEG = 0.002         # ~200m edge overlap to avoid missing edge alerts

# Output directories/files
OUT_DIR = Path("./out")
OUT_DIR.mkdir(exist_ok=True, parents=True)

GRID_STATE_PATH = OUT_DIR / "grid_state.json"
GRID_GEOJSON_PATH = OUT_DIR / "grid.geojson"
ALERTS_GEOJSON_PATH = OUT_DIR / "alerts.geojson"
MAP_HTML_PATH = OUT_DIR / "map.html"

# Timeouts
HTTP_TIMEOUT = 12  # seconds


# ----------------------------- Data structures -----------------------------

@dataclass
class Tile:
    id: str
    left: float
    right: float
    bottom: float
    top: float
    level: int = 0  # 0 = base tiles
    last_count: int = 0
    split_next: bool = False
    # Optional: quiet streaks for merges; not deeply used in this basic script
    quiet_streak: int = 0

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    def bbox_with_overlap(self) -> Tuple[float, float, float, float]:
        return (
            self.left  - OVERLAP_DEG,
            self.right + OVERLAP_DEG,
            self.bottom - OVERLAP_DEG,
            self.top     + OVERLAP_DEG,
        )


# ----------------------------- Utilities -----------------------------

def make_base_grid() -> List[Tile]:
    tiles: List[Tile] = []
    dx = (SYDNEY_RIGHT - SYDNEY_LEFT) / BASE_COLS
    dy = (SYDNEY_TOP - SYDNEY_BOTTOM) / BASE_ROWS
    tid = 0
    for r in range(BASE_ROWS):
        for c in range(BASE_COLS):
            left = SYDNEY_LEFT + c * dx
            right = left + dx
            bottom = SYDNEY_BOTTOM + r * dy
            top = bottom + dy
            tiles.append(Tile(
                id=f"L0_{r}_{c}",
                left=left, right=right, bottom=bottom, top=top, level=0
            ))
            tid += 1
    return tiles

def load_grid_state() -> List[Tile]:
    if GRID_STATE_PATH.exists():
        with open(GRID_STATE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        tiles = [Tile(**t) for t in raw.get("tiles", [])]
        return tiles
    else:
        return make_base_grid()

def save_grid_state(tiles: List[Tile]) -> None:
    # Persist only leaf tiles
    data = {"tiles": [asdict(t) for t in tiles]}
    with open(GRID_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def split_tile(tile: Tile) -> List[Tile]:
    midx = (tile.left + tile.right) / 2.0
    midy = (tile.bottom + tile.top) / 2.0
    lvl = tile.level + 1
    base_id = tile.id

    return [
        Tile(id=f"{base_id}_NW", left=tile.left, right=midx, bottom=midy, top=tile.top, level=lvl),
        Tile(id=f"{base_id}_NE", left=midx, right=tile.right, bottom=midy, top=tile.top, level=lvl),
        Tile(id=f"{base_id}_SW", left=tile.left, right=midx, bottom=tile.bottom, top=midy, level=lvl),
        Tile(id=f"{base_id}_SE", left=midx, right=tile.right, bottom=tile.bottom, top=midy, level=lvl),
    ]

def tile_too_small(tile: Tile) -> bool:
    return tile.width <= MIN_TILE_WIDTH_DEG or tile.height <= MIN_TILE_HEIGHT_DEG


# ----------------------------- Waze API -----------------------------

def query_waze_alerts(left: float, right: float, bottom: float, top: float) -> Dict:
    params = dict(WAZE_PARAMS_TEMPLATE)
    params.update({
        "left": f"{left:.10f}",
        "right": f"{right:.10f}",
        "bottom": f"{bottom:.10f}",
        "top": f"{top:.10f}",
    })
    r = requests.get(WAZE_URL, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    try:
        return r.json()
    except requests.exceptions.JSONDecodeError:
        # some Waze endpoints occasionally return text; try manual load
        return json.loads(r.text)


# ----------------------------- GeoJSON helpers -----------------------------

def tiles_to_geojson(tiles: List[Tile]) -> Dict:
    feats = []
    for t in tiles:
        poly = [
            [t.left, t.bottom],
            [t.right, t.bottom],
            [t.right, t.top],
            [t.left, t.top],
            [t.left, t.bottom],
        ]
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [poly]},
            "properties": {
                "id": t.id,
                "level": t.level,
                "last_count": t.last_count
            }
        })
    return {"type": "FeatureCollection", "features": feats}

def alerts_to_geojson(alerts: List[Dict]) -> Dict:
    feats = []
    for a in alerts:
        loc = a.get("location") or {}
        x = loc.get("x")
        y = loc.get("y")
        if x is None or y is None:
            # fallback if different structure
            continue
        props = {k: v for k, v in a.items() if k != "location"}
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": props
        })
    return {"type": "FeatureCollection", "features": feats}


# ----------------------------- Map writer -----------------------------

LEAFLET_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Waze Hybrid Grid</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map { height: 100%; margin: 0; }
    .legend { background: white; padding: 8px 10px; border-radius: 4px; line-height: 1.3; }
    .legend b { display: block; margin-bottom: 4px; }
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const grid = __GRID_GEOJSON__;
    const alerts = __ALERTS_GEOJSON__;

    const map = L.map('map').setView([-33.87, 151.1], 11);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    function styleByCount(f) {
      const c = f.properties.last_count || 0;
      // simple stepped style
      let weight = 1;
      let dashArray = '2,2';
      let fillOpacity = 0.05;
      let color = '#000';
      if (c >= 150) { color = '#d73027'; fillOpacity = 0.12; weight = 2; dashArray = null; }
      else if (c >= 100) { color = '#fc8d59'; fillOpacity = 0.10; }
      else if (c >= 50)  { color = '#fee08b'; fillOpacity = 0.08; }
      else if (c >= 1)   { color = '#d9ef8b'; fillOpacity = 0.06; }
      return { color, weight, dashArray, fillOpacity };
    }

    const gridLayer = L.geoJSON(grid, {
      style: styleByCount,
      onEachFeature: (f, layer) => {
        layer.bindPopup(`<b>${f.properties.id}</b><br/>level: ${f.properties.level}<br/>count: ${f.properties.last_count}`);
      }
    }).addTo(map);

    const alertLayer = L.geoJSON(alerts, {
      pointToLayer: (f, latlng) => L.circleMarker(latlng, { radius: 4, weight: 1 })
        .bindPopup(`<b>${f.properties.type || 'ALERT'}</b><br/>city: ${f.properties.city || ''}`)
    }).addTo(map);

    const grp = L.featureGroup([gridLayer, alertLayer]);
    map.fitBounds(grp.getBounds().pad(0.1));

    // legend
    const legend = L.control({position:'topright'});
    legend.onAdd = function (map) {
      const div = L.DomUtil.create('div', 'legend');
      div.innerHTML = '<b>Tile counts</b>' +
        '<div><span style="background:#d73027;color:#fff;padding:2px 4px;border-radius:2px">≥150</span> split</div>' +
        '<div><span style="background:#fc8d59;padding:2px 4px;border-radius:2px">100–149</span></div>' +
        '<div><span style="background:#fee08b;padding:2px 4px;border-radius:2px">50–99</span></div>' +
        '<div><span style="background:#d9ef8b;padding:2px 4px;border-radius:2px">1–49</span></div>';
      return div;
    };
    legend.addTo(map);
  </script>
</body>
</html>
"""

def write_map_html(grid_geojson: Dict, alerts_geojson: Dict, path: Path) -> None:
    html = LEAFLET_HTML.replace("__GRID_GEOJSON__", json.dumps(grid_geojson))\
                       .replace("__ALERTS_GEOJSON__", json.dumps(alerts_geojson))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ----------------------------- Core cycle -----------------------------

def poll_once(tiles: List[Tile], multi_cycle: bool = False) -> Tuple[List[Tile], List[Dict], bool]:
    next_tiles: List[Tile] = []
    all_alerts: List[Dict] = []
    any_splits = False

    for i, t in enumerate(tiles, 1):
        # Skip tiles that were previously quiet in multi-cycle mode
        if multi_cycle and t.last_count < SPLIT_THRESHOLD and t.level > 0:
            print(f"[{i}/{len(tiles)}] Skipping quiet tile {t.id} (level {t.level}, last count: {t.last_count})")
            next_tiles.append(t)
            continue

        left, right, bottom, top = t.bbox_with_overlap()
        print(f"[{i}/{len(tiles)}] Polling tile {t.id} (level {t.level})...", end="", flush=True)
        
        try:
            data = query_waze_alerts(left, right, bottom, top)
            alerts = data.get("alerts", []) if isinstance(data, dict) else []
            count = len(alerts)
            print(f" {count} alerts found")
            
        except Exception as e:
            print(f" ERROR: {e}", file=sys.stderr)
            alerts, count = [], 0

        t.last_count = count
        # Decide split for NEXT run (not immediate, to avoid thrash)
        if count >= SPLIT_THRESHOLD and not tile_too_small(t):
            t.split_next = True
            any_splits = True
            print(f"    → Will split next cycle (count >= {SPLIT_THRESHOLD})")
        else:
            t.split_next = False

        # Accumulate alerts (no de-dupe at this stage)
        all_alerts.extend(alerts)

        next_tiles.append(t)

    # After polling all tiles, build the tile set for the next run based on split flags
    refined_tiles: List[Tile] = []
    for t in next_tiles:
        if t.split_next and not tile_too_small(t):
            refined_tiles.extend(split_tile(t))
        else:
            refined_tiles.append(t)

    return refined_tiles, all_alerts, any_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=1, help="Number of polling cycles to run (default 1)")
    parser.add_argument("--sleep-sec", type=int, default=0, help="Sleep seconds between cycles (default 0)")
    parser.add_argument("--reset-grid", action="store_true", help="Ignore saved grid and start from base grid")
    args = parser.parse_args()

    tiles = make_base_grid() if args.reset_grid else load_grid_state()

    cycle = 0
    while cycle < args.cycles:
        print(f"Cycle {cycle+1}/{args.cycles}: polling {len(tiles)} tiles...")
        tiles, alerts, any_splits = poll_once(tiles, multi_cycle=args.cycles > 1)

        # Write outputs of this cycle
        grid_geojson = tiles_to_geojson(tiles)
        alerts_geojson = alerts_to_geojson(alerts)

        with open(GRID_GEOJSON_PATH, "w", encoding="utf-8") as f:
            json.dump(grid_geojson, f, indent=2)
        with open(ALERTS_GEOJSON_PATH, "w", encoding="utf-8") as f:
            json.dump(alerts_geojson, f, indent=2)

        write_map_html(grid_geojson, alerts_geojson, MAP_HTML_PATH)

        save_grid_state(tiles)
        print(f"  Wrote {GRID_GEOJSON_PATH}, {ALERTS_GEOJSON_PATH}, {MAP_HTML_PATH}")
        
        # If no splits occurred and we're in multi-cycle mode, we can stop early
        if not any_splits and args.cycles > 1:
            print(f"No splits needed - stopping after {cycle+1} cycles")
            break
            
        cycle += 1
        if cycle < args.cycles and args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    print("Done. Open ./out/map.html in a browser to view.")


if __name__ == "__main__":
    main()
