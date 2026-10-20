"""The README's maps: the GL model's value of one standard home in every census block, over an OpenStreetMap basemap,
with a zoom on one township. Two sets of fits:
  --set county2025 (public/figures/location_map.png): the whole-county fits of all 38 townships, Chicago included,
    on the Assessor's 2025 training sales, valued at October 2023 (their last training month);
  --set assessment (public/figures/location_map_assessment.png): the 1 January fits of the 30 suburban townships
    (north 2022, south and west 2023).

Value of a block = posterior median, over the 4,000 draws of the published model P, of the log value in the last
training month of the same standard home everywhere: intercept + mu[last training month] + zeta[k] + psi[k, last
month] + u[s] + (x_std - training mean)' beta (stan/v2/gl_struct.stan; value_v2.py's formula). x_std is the median of
each numeric characteristic (the previous-sale count included, where it is a regressor) and the most common level of
each categorical one over the training sales of all the set's townships, coded with each township's own coding. Shown
as exp(difference) - 1 in % from the median block shown, on one colour scale for the whole county, so colours compare
across township lines. Blocks without eligible homes are left blank. A block that straddles two Assessor
neighbourhoods is one unit per neighbourhood in the model; the map shows their mean weighted by homes. The Chicago
outline is the Assessor's eight City townships (70-77).

Inputs: the set's runs.json (public/runs/v2/county2025_eval/ or struct_eval/, model P), each run's draws.npz and
design.npz, the prepared units (public/data/v2/prep_*/t<code>/units.parquet), the census-block units of each year
(public/data/v2/blocks_2023_units.parquet, ay2022/blocks_2022_units.parquet) and the 2020 TIGER/Line block file
(notes/v2/explore_blocks/tl_2020_17_tabblock20.zip). Basemap tiles are downloaded once from tile.openstreetmap.org
(cached in public/data/v2/osm_tiles/); map data (c) OpenStreetMap contributors, ODbL.

Usage (from the working folder): .venv/bin/python -I public/scripts/v2/fig_location_map.py
[--set county2025|assessment] [--style colour|light] [--zoom-town 17] [--out <png>]
"""
import argparse
import io
import json
import math
import time
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

REPO = Path(__file__).resolve().parents[3]
TIGER = REPO / "notes/v2/explore_blocks/tl_2020_17_tabblock20.zip"
TILES = REPO / "public/data/v2/osm_tiles"
UNITS = {"2023": REPO / "public/data/v2/blocks_2023_units.parquet",
         "2022": REPO / "public/data/v2/ay2022/blocks_2022_units.parquet"}
USER_AGENT = "gl-avm-cook-figure/1.0 (research figure; github.com/wsijp)"   # OSM's tile policy asks for one
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
NAMES = {"70": "Hyde Park", "71": "Jefferson", "72": "Lake", "73": "Lake View", "74": "North Chicago",
         "75": "Rogers Park", "76": "South Chicago", "77": "West Chicago", "10": "Barrington", "11": "Berwyn", "12": "Bloom", "13": "Bremen", "14": "Calumet", "15": "Cicero",
         "16": "Elk Grove", "17": "Evanston", "18": "Hanover", "19": "Lemont", "20": "Leyden", "21": "Lyons",
         "22": "Maine", "23": "New Trier", "24": "Niles", "25": "Northfield", "26": "Norwood Park",
         "27": "Oak Park", "28": "Orland", "29": "Palatine", "30": "Palos", "31": "Proviso", "32": "Rich",
         "33": "River Forest", "34": "Riverside", "35": "Schaumburg", "36": "Stickney", "37": "Thornton",
         "38": "Wheeling", "39": "Worth"}
CITY = {str(t) for t in range(70, 78)}
# blue (below the median block) to red (above it) through a neutral grey
CMAP = LinearSegmentedColormap.from_list("div", ["#1f4e9c", "#6fa0dc", "#f0efec", "#e98a6b", "#b2241b"])
# colour scale in log points, about the 95th (assessment: +-0.75, -53% to +112%) or 97th (county: +-1.0, -63% to
# +172%) percentile of |difference|; beyond is saturated. Ticks in % of price
LIMIT = {"assessment": 0.75, "county2025": 1.0}
TICKS = {"assessment": [-50, -25, 0, 50, 100], "county2025": [-50, -25, 0, 50, 100]}


def decode(t, coding):
    """characteristics back from one township's coded regressors: numerics on their raw scale (missing values were
    filled with the training median), categoricals as levels ("other" pools the township's rare levels); characteristics
    the township's coding dropped are absent"""
    x = pd.DataFrame(index=t.index)
    for c, p in coding["numeric"].items():
        if f"x:{c}" in t:
            x[c] = t[f"x:{c}"] * p["sd"] + p["mean"]
    if "x:char_recent_renovation" in t:
        x["char_recent_renovation"] = t["x:char_recent_renovation"]
    for c, p in coding["categorical"].items():
        s = pd.Series(p["reference"], index=t.index)
        for lev in p["levels"]:
            if f"x:{c}={lev}" in t:
                s[t[f"x:{c}={lev}"] == 1] = lev
        x[c] = s
    return x


def standard_home(runs):
    """the standard home: medians of the numeric characteristics and the most common level of each categorical one,
    over the training sales of all the fitted townships (decoded from each township's coded regressors)"""
    xs = []
    for r in runs:
        prep = REPO / r["prep"] / f"t{r['town']}"
        xs.append(decode(pd.read_parquet(prep / "train.parquet"), json.loads((prep / "coding.json").read_text())))
    x = pd.concat(xs)
    cat = sorted({c for r in runs for c in json.loads((REPO / r["prep"] / f"t{r['town']}" / "coding.json").read_text())
                  ["categorical"]})
    num = [c for c in x.columns if c not in cat]
    return ({c: float(x[c].median()) for c in num},
            {c: x.loc[x[c] != "other", c].value_counts().index[0] for c in cat})   # "other" differs by township


def encode(home, coding, xcols):
    """the standard home in one township's coded regressors (prep_v2.apply_coding's rules)"""
    num, cat = home
    x = {}
    for c, p in coding["numeric"].items():
        x[f"x:{c}"] = (num[c] - p["mean"]) / p["sd"] if p["sd"] > 0 else 0.0
    x["x:char_recent_renovation"] = num.get("char_recent_renovation", 0.0)
    for c, p in coding["categorical"].items():
        lev = "other" if cat[c] in p["merged_into_other"] else cat[c]
        if lev not in p["levels"]:
            lev = "other" if "other" in p["levels"] else p["reference"]
        for l in p["levels"]:
            if l != p["reference"]:
                x[f"x:{c}={l}"] = float(lev == l)
    return np.array([x[c] for c in xcols])


def unit_values(r, home):
    """one fit's units (spatial nodes) with the posterior median 1 January log value of the standard home there"""
    prep = REPO / r["prep"] / f"t{r['town']}"
    d = np.load(REPO / r["run"] / "draws.npz")
    design = np.load(REPO / r["run"] / "design.npz")
    nbhds, xcols = list(design["nbhds"]), list(design["xcols"])
    units = pd.read_parquet(prep / "units.parquet")
    assert (units["node"].to_numpy() == np.arange(len(units))).all() and d["u"].shape[1] == len(units)
    xmean = pd.read_parquet(prep / "train.parquet", columns=xcols).mean().to_numpy()   # design.npz's X_c centring
    x_std = encode(home, json.loads((prep / "coding.json").read_text()), xcols) - xmean
    k = units["nbhd_id"].map({n: i for i, n in enumerate(nbhds)}).to_numpy()
    common = d["intercept"] + d["mu"][:, -1] + d["beta"] @ x_std
    return units.assign(value=np.median(common[:, None] + d["zeta"][:, k] + d["psi_last"][:, k] + d["u"], axis=0))


def published_runs(which):
    if which == "county2025":                        # one fit per township, all on the 2023 census-block units
        return [dict(r, model="P", year="2023") for r in
                json.loads((REPO / "public/runs/v2/county2025_eval/runs.json").read_text()).values()]
    return [r for r in json.loads((REPO / "public/runs/v2/struct_eval/runs.json").read_text()).values()
            if r["model"] == "P"]


def block_values(which):
    """one row per census block: the model's log value of the standard home there in the last training month
    (posterior median of intercept + mu[last month] + zeta[k] + psi[k, last month] + u[s] + (x_std - training mean)'
    beta); the fitted townships; their years; the standard home"""
    runs = published_runs(which)
    home = standard_home(runs)
    out = []
    for r in sorted(runs, key=lambda r: (r["year"], r["town"])):
        year, town = r["year"], r["town"]
        units = unit_values(r, home)
        u = units[~units["pass_through"]]
        geoid = pd.read_parquet(UNITS[year], columns=["unit_idx", "block_geoid"]).set_index("unit_idx")["block_geoid"]
        u = u.assign(block_geoid=u["unit_idx"].map(geoid), year=year, town=town)
        assert u["block_geoid"].notna().all()
        out.append(u[["year", "town", "block_geoid", "value", "n_supported"]])
    e = pd.concat(out)
    w = e["n_supported"].clip(lower=1)
    towns = set(e["town"])
    # a block split between two neighbourhoods (or, for 12 blocks, two fitted townships of the same year) is one unit
    # per part in the model: the map shows their mean weighted by homes
    e = e.assign(wx=e["value"] * w, w=w).groupby(["year", "block_geoid"], as_index=False)[["wx", "w"]].sum()
    e["value"] = e["wx"] / e["w"]
    assert e["block_geoid"].is_unique, "a block in both years"
    e["effect"] = e["value"] - e["value"].median()             # log difference from the median block shown
    return e[["block_geoid", "value", "effect"]], towns, {r["town"]: r["year"] for r in runs}, home


def tile(z, x, y):
    f = TILES / f"{z}/{x}/{y}.png"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            f.write_bytes(resp.read())
        time.sleep(0.1)
    return plt.imread(io.BytesIO(f.read_bytes()), format="png")[..., :3]


def basemap(ax, bounds, z, style):
    """draw the OSM tiles covering bounds (EPSG:3857) at zoom z; 'light' washes them out to grey"""
    R = 6378137.0
    def to_tile(mx, my):
        lon, lat = math.degrees(mx / R), math.degrees(2 * math.atan(math.exp(my / R)) - math.pi / 2)
        n = 2 ** z
        return (int((lon + 180) / 360 * n),
                int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n))
    x0, y1 = to_tile(bounds[0], bounds[1])
    x1, y0 = to_tile(bounds[2], bounds[3])
    img = np.vstack([np.hstack([tile(z, x, y) for x in range(x0, x1 + 1)]) for y in range(y0, y1 + 1)])
    if style == "light":
        grey = img @ np.array([0.299, 0.587, 0.114])
        img = np.repeat((0.55 + 0.45 * grey)[..., None], 3, axis=2)
    else:
        img = 0.3 + 0.7 * img                        # slightly faded, so the blocks stand out
    size = 2 * math.pi * R / 2 ** z
    ext = (-math.pi * R + x0 * size, -math.pi * R + (x1 + 1) * size,
           math.pi * R - (y1 + 1) * size, math.pi * R - y0 * size)
    ax.imshow(img, extent=ext, interpolation="lanczos", zorder=0)


def outline(geom):
    """the outer boundary of a set of land blocks: rivers and canals between blocks closed, holes dropped"""
    from shapely.geometry import MultiPolygon, Polygon
    g = geom.buffer(250).buffer(-250)
    parts = g.geoms if isinstance(g, MultiPolygon) else [g]
    parts = [Polygon(q.exterior) for q in parts if q.area > 5e6]
    return gpd.GeoSeries([MultiPolygon(parts) if len(parts) > 1 else parts[0]], crs=3857).simplify(60)


def halo(width=2.5):
    return [pe.Stroke(linewidth=width, foreground="white"), pe.Normal()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", choices=["county2025", "assessment"], default="county2025")
    ap.add_argument("--style", choices=["colour", "light"], default="colour")
    ap.add_argument("--zoom-town", default=None)       # county2025: West Chicago (77); assessment: Evanston (17)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    county_set = a.which == "county2025"
    a.zoom_town = a.zoom_town or ("77" if county_set else "17")
    a.out = a.out or ("public/figures/location_map.png" if county_set else "public/figures/location_map_assessment.png")

    e, fitted_towns, year_of, home = block_values(a.which)
    print("standard home:", home)
    print(f"median block: ${math.exp(e['value'].median()):,.0f}")
    blk = gpd.read_file(f"zip://{TIGER}", where="COUNTYFP20 = '031'", engine="pyogrio",
                        columns=["COUNTYFP20", "GEOID20", "ALAND20"]).to_crs(3857)
    towns_of = pd.read_parquet(UNITS["2023"], columns=["block_geoid", "town"]).drop_duplicates("block_geoid")
    land = blk[blk["ALAND20"] > 0].merge(towns_of, left_on="GEOID20", right_on="block_geoid")
    towns = land.dissolve("town")[["geometry"]]
    towns["geometry"] = towns.buffer(30).buffer(-30).simplify(40)
    county = outline(towns.union_all())
    chicago = outline(towns[towns.index.isin(CITY)].union_all())
    fitted = towns[towns.index.isin(fitted_towns)]
    m = blk.merge(e, left_on="GEOID20", right_on="block_geoid")
    print(f"{len(m):,} blocks with an effect in {len(fitted_towns)} townships; "
          f"{(m['effect'].abs() > LIMIT[a.which]).mean():.1%} beyond the colour scale")

    plt.rcParams.update({"font.size": 12, "text.color": INK, "figure.facecolor": SURFACE})
    norm = Normalize(-LIMIT[a.which], LIMIT[a.which])
    fig = plt.figure(figsize=(12, 9.2))
    ax = fig.add_axes([0.0, 0.08, 0.56, 0.92])
    zx = fig.add_axes([0.57, 0.24, 0.42, 0.72])

    # county panel
    b = county.total_bounds
    pad = 1500
    bounds = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
    basemap(ax, bounds, 11, a.style)
    m.plot(ax=ax, column="effect", cmap=CMAP, norm=norm, linewidth=0, zorder=2)
    county.boundary.plot(ax=ax, color=INK, linewidth=1.4, zorder=5)
    if county_set:                                   # thin light township lines; the City outline a little stronger
        fitted.boundary.plot(ax=ax, color="white", linewidth=0.45, alpha=0.8, zorder=4)
        chicago.boundary.plot(ax=ax, color=INK2, linewidth=0.9, linestyle=(0, (4, 2)), zorder=5)
        c = chicago.iloc[0].representative_point()
        ax.annotate("Chicago", (c.x + 1500, c.y - 15000), ha="center", va="center", fontsize=13, fontweight="bold",
                    color=INK, zorder=6, path_effects=halo(3))
    else:
        fitted.boundary.plot(ax=ax, color="#3a3a38", linewidth=0.6, zorder=4)
        chicago.plot(ax=ax, color="#d9d7d0", alpha=0.45, linewidth=0, zorder=1)
        chicago.boundary.plot(ax=ax, color=INK2, linewidth=1.0, linestyle=(0, (4, 2)), zorder=5)
        c = chicago.iloc[0].representative_point()
        ax.annotate("Chicago\n(not covered)", (c.x + 2500, c.y + 4000), ha="center", va="center", fontsize=13,
                    color=INK2, style="italic", zorder=6, path_effects=halo(3))
        for lab, t in [("North suburbs, 2022", "25"), ("South and west suburbs, 2023", "13")]:
            p = towns.loc[t].geometry.representative_point()
            ax.annotate(lab, (p.x, p.y), ha="center", va="center", fontsize=13, fontweight="bold",
                        color=INK, zorder=6, path_effects=halo(3))
    zt = towns.loc[a.zoom_town].geometry
    zb = zt.bounds
    zpad = 900
    zbounds = (zb[0] - zpad, zb[1] - zpad, zb[2] + zpad, zb[3] + zpad)
    ax.add_patch(plt.Rectangle(zbounds[:2], zbounds[2] - zbounds[0], zbounds[3] - zbounds[1], fill=False,
                               edgecolor=INK, linewidth=1.6, zorder=7))
    ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal"); ax.axis("off")

    # zoom panel
    w, h = zbounds[2] - zbounds[0], zbounds[3] - zbounds[1]
    ratio = 0.72 * 9.2 / (0.42 * 12)                 # panel height / width
    if h / w < ratio:
        extra = (w * ratio - h) / 2
        zbounds = (zbounds[0], zbounds[1] - extra, zbounds[2], zbounds[3] + extra)
    else:
        extra = (h / ratio - w) / 2
        zbounds = (zbounds[0] - extra, zbounds[1], zbounds[2] + extra, zbounds[3])
    basemap(zx, zbounds, 14, a.style)
    m.plot(ax=zx, column="effect", cmap=CMAP, norm=norm, linewidth=0.15, edgecolor="white", alpha=0.85, zorder=2)
    gpd.GeoSeries([zt], crs=3857).boundary.plot(ax=zx, color=INK, linewidth=1.2, zorder=4)
    zx.set_xlim(zbounds[0], zbounds[2]); zx.set_ylim(zbounds[1], zbounds[3])
    zx.set_aspect("equal"); zx.set_xticks([]); zx.set_yticks([]); zx.set_xlabel(""); zx.set_ylabel("")
    for s in zx.spines.values():
        s.set_edgecolor(INK); s.set_linewidth(1.6)
    zlab = NAMES[a.zoom_town] if county_set else f"{NAMES[a.zoom_town]} ({year_of[a.zoom_town]})"
    zx.set_title(f"Zoom: {zlab}, one colour per census block", fontsize=13, loc="left")

    # legend
    cax = fig.add_axes([0.60, 0.10, 0.36, 0.026])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([math.log1p(t / 100) for t in TICKS[a.which]],
                labels=[f"{t:+d}%" if t else "0" for t in TICKS[a.which]])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=12, length=3, color=INK2)
    when = "in October 2023" if county_set else "at 1 January"
    cb.set_label(f"Value of one standard home {when},\nrelative to the median block", fontsize=13, labelpad=8)
    cb.ax.xaxis.set_label_position("top")
    fig.text(0.995, 0.005, "Map data © OpenStreetMap contributors (ODbL)", fontsize=10, color=INK2, ha="right")
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    from PIL import Image                            # 256 colours: a third of the file size, no visible change
    Image.open(out).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT,
                                            dither=Image.Dither.NONE).save(out, optimize=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
