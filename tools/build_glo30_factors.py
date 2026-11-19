"""Generate the GLO-30 landform and hydrology rasters from a prepared 30 m DEM."""

import argparse
from collections import deque
import heapq
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

RES = 30.0


def build(dem_path, output_dir):
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError("Use a dedicated empty directory for the new rasters.")
    else:
        output_dir.mkdir(parents=True)

    def save_factor(arr, name, profile, nodata, meta_entry):
        out = np.where(np.isfinite(arr), arr, nodata)
        with rasterio.open(output_dir / f"{name}.tif", "w", **profile) as dst:
            dst.write(out.astype("float32"), 1)

    with rasterio.open(dem_path) as src:
        if src.count != 1 or src.crs is None or not src.crs.is_projected or not np.allclose(src.res, (30, 30)):
            raise ValueError("Expected the 30 m projected, single-band GLO-30 working DEM")
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        h, w = dem.shape
    print(f"DEM shape {h}x{w}, transform {transform}, crs {crs}")

    dem[dem <= -32000.0] = np.nan
    valid = ~np.isnan(dem)

    out_prof = dict(driver="GTiff", height=h, width=w, count=1, dtype="float32",
                    crs=crs, transform=transform, nodata=-9999.0, compress="deflate")
    NODATA = -9999.0

    # ---------------------------------------------------------------------------
    # 2. slope + aspect (central difference)
    # ---------------------------------------------------------------------------
    gy, gx = np.gradient(dem, RES, RES)   # gy = d/drow (south+), gx = d/dcol (east+)
    gy = np.where(valid, gy, np.nan)
    gx = np.where(valid, gx, np.nan)
    slope_rad = np.arctan(np.hypot(gx, gy))
    slope_tan = np.tan(slope_rad)          # rise/run
    aspect = np.arctan2(gx, -gy)           # 0=north, clockwise positive
    northness = np.cos(aspect)
    eastness = np.sin(aspect)

    # ---------------------------------------------------------------------------
    # 3. geometric / morphometric factors (focal window)
    # ---------------------------------------------------------------------------
    def focal_mean(a, m):
        return ndi.uniform_filter(a, size=m, mode="nearest")

    def focal_max(a, m):
        return ndi.maximum_filter(a, size=m, mode="nearest")

    def focal_min(a, m):
        return ndi.minimum_filter(a, size=m, mode="nearest")

    def focal_std(a, m):
        mean = ndi.uniform_filter(a, size=m, mode="nearest")
        sq = ndi.uniform_filter(a * a, size=m, mode="nearest")
        return np.sqrt(np.maximum(sq - mean * mean, 0))

    # TPI at 90/270/900 m (window in cells)
    for label, meters in [("tpi90", 90), ("tpi270", 270), ("tpi900", 900)]:
        r = max(1, int(round(meters / RES)))
        size = 2 * r + 1
        tpi = dem - focal_mean(dem, size)
        save_factor(tpi, f"LFORM_TPI_{meters}m", out_prof, NODATA,
                    {"factor": f"LFORM_TPI_{meters}m", "method": "z - focal_mean",
                     "window_m": meters, "module": "LANDFORM"})
        print(f"  TPI {meters}m done (window {size}x{size})")

    # TRI (3x3 mean absolute difference to 8 neighbors)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    tri = np.abs(ndi.correlate(dem, k) - 8 * dem) / 8.0
    save_factor(tri, "LFORM_TRI", out_prof, NODATA,
                {"factor": "LFORM_TRI", "method": "mean|z-8neighbors| (3x3)", "module": "LANDFORM"})
    print("  TRI done")

    # roughness (focal std @90m)
    r = int(round(90 / RES)); size = 2 * r + 1
    rough = focal_std(dem, size)
    save_factor(rough, "LFORM_roughness_90m", out_prof, NODATA,
                {"factor": "LFORM_roughness_90m", "method": "focal_std", "window_m": 90, "module": "LANDFORM"})
    print("  roughness done")

    # local relief 270m / elevation range 900m
    for label, meters in [("270", 270), ("900", 900)]:
        r = int(round(meters / RES)); size = 2 * r + 1
        relief = focal_max(dem, size) - focal_min(dem, size)
        save_factor(relief, f"LFORM_relief_{meters}m", out_prof, NODATA,
                    {"factor": f"LFORM_relief_{meters}m", "method": "focal max-min",
                     "window_m": meters, "module": "LANDFORM"})
        print(f"  relief {meters}m done")

    # mean slope 270m
    r = int(round(270 / RES)); size = 2 * r + 1
    mean_slope = focal_mean(slope_tan, size) * 100.0
    save_factor(mean_slope, "LFORM_mean_slope_pct_270m", out_prof, NODATA,
                {"factor": "LFORM_mean_slope_pct_270m", "method": "focal mean of tan(slope)*100",
                 "window_m": 270, "module": "LANDFORM"})
    print("  mean slope 270m done")

    # northness / eastness
    save_factor(northness, "LFORM_northness", out_prof, NODATA,
                {"factor": "LFORM_northness", "method": "cos(aspect)", "module": "LANDFORM"})
    save_factor(eastness, "LFORM_eastness", out_prof, NODATA,
                {"factor": "LFORM_eastness", "method": "sin(aspect)", "module": "LANDFORM"})
    print("  northness/eastness done")

    # valley depth 270m (ridge max - cell) ; RSP 270m (z-min)/(max-min)
    r = int(round(270 / RES)); size = 2 * r + 1
    fmax = focal_max(dem, size)
    fmin = focal_min(dem, size)
    valley_depth = fmax - dem
    rsp = np.where((fmax - fmin) > 0, (dem - fmin) / np.maximum(fmax - fmin, 1e-6), np.nan)
    save_factor(valley_depth, "HYDRO_valley_depth_270m", out_prof, NODATA,
                {"factor": "HYDRO_valley_depth_270m", "method": "focal_max - z", "window_m": 270, "module": "HYDRO"})
    save_factor(rsp, "HYDRO_rsp_270m", out_prof, NODATA,
                {"factor": "HYDRO_rsp_270m", "method": "(z-focal_min)/(focal_max-focal_min)",
                 "window_m": 270, "module": "HYDRO"})
    print("  valley depth / RSP done")

    # ---------------------------------------------------------------------------
    # 4. hydrology: fill -> D8 -> flat resolution -> accumulation
    # ---------------------------------------------------------------------------
    print("\n[hydrology] sink fill ...")
    # use a filled working array (float64 for stability)
    fillval = np.where(valid, dem, -9999.0).astype(np.float64)
    visited = np.zeros((h, w), dtype=bool)
    heap = []
    # seed = boundary valid cells
    for i in range(h):
        for j in (0, w - 1):
            if valid[i, j]:
                visited[i, j] = True
                heapq.heappush(heap, (fillval[i, j], i, j))
    for j in range(w):
        for i in (0, h - 1):
            if valid[i, j] and not visited[i, j]:
                visited[i, j] = True
                heapq.heappush(heap, (fillval[i, j], i, j))

    offs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while heap:
        e, i, j = heapq.heappop(heap)
        for di, dj in offs:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and valid[ni, nj] and not visited[ni, nj]:
                visited[ni, nj] = True
                if fillval[ni, nj] < e:
                    fillval[ni, nj] = e
                heapq.heappush(heap, (fillval[ni, nj], ni, nj))
    filled = np.round(fillval, 2)  # round to 1 cm: remove float32 quantization noise
    print(f"  fill done (any unvisited valid cells: {int(valid.sum()) - int(visited.sum())})")

    # D8 direction (steepest descent); distances 1 (cardinal) / sqrt2 (diagonal)
    d8_di = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    d8_dist = np.array([1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)])

    def src_tgt_slices(di, dj):
        si0 = max(0, -di); si1 = h - max(0, di)
        sj0 = max(0, -dj); sj1 = w - max(0, dj)
        return (slice(si0, si1), slice(sj0, sj1)), (slice(si0 + di, si1 + di), slice(sj0 + dj, sj1 + dj))

    # choose max drop among 8 neighbors; STRICTLY downhill only (drop > 0).
    # A cell with no lower valid neighbor stays d8 == -1 (outlet or flat).
    best_drop = np.full((h, w), -np.inf)
    d8 = np.full((h, w), -1, dtype=np.int8)
    for k in range(8):
        di, dj = d8_di[k]
        src, tgt = src_tgt_slices(di, dj)
        drop = np.where(valid[tgt], (filled[src] - filled[tgt]) / d8_dist[k], -np.inf)
        upd = (drop > best_drop[src]) & (drop > 0.0)
        best_drop[src][upd] = drop[upd]
        d8[src][upd] = k

    # flat resolution: cells with no downhill (d8==-1) route to an exactly-equal
    # neighbor that is already resolved. Uses a per-iteration snapshot of d8 so a
    # cell can only point to a cell resolved in an EARLIER iteration -> strictly
    # acyclic BFS outward from the resolved boundary (no 2-cycles).
    print("  flat resolution ...")
    unresolved = (d8 == -1) & valid
    MAX_FLAT = 3000
    for _ in range(MAX_FLAT):
        d8_snap = d8.copy()
        changed = False
        for k in range(8):
            di, dj = d8_di[k]
            src, tgt = src_tgt_slices(di, dj)
            # exact equality (post 1 cm rounding) — consistent with D8's drop>0,
            # so no D8 edge can be reversed here (guarantees acyclicity)
            same = filled[src] == filled[tgt]
            tgt_ok = (d8_snap[tgt] != -1) & valid[tgt]
            m = unresolved[src] & same & tgt_ok
            if m.any():
                d8[src][m] = k
                unresolved[src] &= ~m
                changed = True
        if not changed:
            break
    print(f"  flat resolution residual unresolved: {int(unresolved.sum())}")

    # flow accumulation (Kahn's topological order — correct for flat-cell routing)
    print("  flow accumulation (topological) ...")
    # in-degree: number of valid cells draining INTO each cell
    indeg = np.zeros((h, w), dtype=np.int32)
    for k in range(8):
        di, dj = d8_di[k]
        src, tgt = src_tgt_slices(di, dj)
        m = (d8[src] == k) & valid[src] & valid[tgt]
        if not m.any():
            continue
        si0 = max(0, -di); sj0 = max(0, -dj)
        r_idx, c_idx = np.nonzero(m)
        tr = si0 + r_idx + di
        tc = sj0 + c_idx + dj
        np.add.at(indeg, (tr, tc), 1)

    accum = np.ones((h, w), dtype=np.float64)
    accum[~valid] = 0.0
    n_valid_pre = int(valid.sum())
    topo = np.empty(n_valid_pre, dtype=np.int64)  # topological order (sources -> sinks)
    q = deque(np.flatnonzero((indeg == 0) & valid).tolist())
    n_processed = 0
    while q:
        idx = q.popleft()
        topo[n_processed] = idx
        n_processed += 1
        i, j = divmod(idx, w)
        kdir = d8[i, j]
        if kdir < 0:
            continue
        di, dj = d8_di[kdir]
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w and valid[ni, nj]:
            accum[ni, nj] += accum[i, j]
            indeg[ni, nj] -= 1
            if indeg[ni, nj] == 0:
                q.append(ni * w + nj)

    # ---- validation ---------------------------------------------------------
    # Conservation: sum of accum over outlet cells == N_valid (each cell's unit
    # flow drains to exactly one outlet). n_processed == N_valid means no cycles.
    outlet = (d8 == -1) & valid
    n_valid = int(valid.sum())
    n_outlet = int(outlet.sum())
    outlet_flow = float(accum[outlet].sum())
    total_flow = float(accum.sum())
    print(f"  accumulation done: total_flow={total_flow:.1f} N_valid={n_valid} "
          f"n_processed={n_processed} n_outlet={n_outlet} "
          f"outlet_flow={outlet_flow:.1f} (should ~= {n_valid})")
    # sanity: min drop along routed edges (negative => uphill routing bug)
    min_drop = np.inf
    for k in range(8):
        di, dj = d8_di[k]
        src, tgt = src_tgt_slices(di, dj)
        m = (d8[src] == k) & valid[src] & valid[tgt]
        if m.any():
            min_drop = min(min_drop, float((filled[src][m] - filled[tgt][m]).min()))
    print(f"  min drop along routing edges = {min_drop:.4f} m (>=0 means no uphill routing)")

    flow_acc = accum
    # specific catchment area (m2/m) = flow_acc * res ; TWI/SPI
    sca = flow_acc * RES
    slope_eps = np.maximum(slope_tan, 1e-6)
    twi = np.log(sca / slope_eps)
    spi = np.log(sca * slope_eps)
    save_factor(twi, "HYDRO_TWI", out_prof, NODATA,
                {"factor": "HYDRO_TWI", "method": "ln(sca/tan(beta)), sca=flow_acc*30m",
                 "module": "HYDRO"})
    save_factor(spi, "HYDRO_SPI", out_prof, NODATA,
                {"factor": "HYDRO_SPI", "method": "ln(sca*tan(beta))", "module": "HYDRO"})
    save_factor(np.log1p(flow_acc), "HYDRO_log_flowacc", out_prof, NODATA,
                {"factor": "HYDRO_log_flowacc", "method": "log1p(flow accumulation)", "module": "HYDRO"})
    print("  TWI/SPI/logflowacc done")

    # LS factor (RUSLE, Moore & Burch simplified)
    beta = slope_rad
    m = 0.4
    L = ((flow_acc * RES) / 22.13) ** m
    S = np.where(slope_tan < 0.09, 10.8 * np.sin(beta) + 0.03, 16.8 * np.sin(beta) - 0.5)
    LS = L * S
    save_factor(LS, "HYDRO_LS", out_prof, NODATA,
                {"factor": "HYDRO_LS", "method": "RUSLE L=((acc*30)/22.13)^0.4 * S (Moore&Burch)",
                 "module": "HYDRO"})
    print("  LS done")

    # channel threshold (fixed, documented, NOT label-tuned): >= 1000 cells = 0.9 km2
    CH_THRESH = 1000
    channel = flow_acc >= CH_THRESH
    print(f"  channel threshold = {CH_THRESH} cells ({CH_THRESH*RES*RES/1e6:.2f} km2)")

    # distance to stream (scipy distance transform)
    dist = ndi.distance_transform_edt(~(channel & valid), sampling=RES)
    save_factor(dist, "HYDRO_dist_stream", out_prof, NODATA,
                {"factor": "HYDRO_dist_stream", "method": f"edt to channel (acc>={CH_THRESH})",
                 "module": "HYDRO"})
    print("  distance-to-stream done")

    # drainage density (channel length per km2 within 900m window)
    r = int(round(900 / RES)); size = 2 * r + 1
    # approximate: count channel cells in window / window area
    chan_win = ndi.uniform_filter((channel & valid).astype(np.float64), size=size, mode="nearest")
    drain_dens = chan_win * 1e6 / (RES * RES)  # channels per km2 (cell-count proxy)
    save_factor(drain_dens, "HYDRO_drain_density_900m", out_prof, NODATA,
                {"factor": "HYDRO_drain_density_900m", "method": "channel-cell density per km2 (900m window)",
                 "window_m": 900, "module": "HYDRO"})
    print("  drainage density done")

    # HAND (height above nearest drainage): reverse topological order (sinks -> sources)
    print("  HAND ...")
    hand = np.full((h, w), np.nan, dtype=np.float64)
    hand[channel & valid] = 0.0
    for t in range(n_processed - 1, -1, -1):
        idx = int(topo[t])
        i, j = divmod(idx, w)
        if channel[i, j]:
            continue  # already 0
        kdir = d8[i, j]
        if kdir < 0:
            hand[i, j] = 0.0  # outlet that is not a channel
            continue
        di, dj = d8_di[kdir]
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w and valid[ni, nj] and not np.isnan(hand[ni, nj]):
            hand[i, j] = (filled[i, j] - filled[ni, nj]) + hand[ni, nj]
        else:
            hand[i, j] = 0.0
    save_factor(hand.astype(np.float32), "HYDRO_HAND", out_prof, NODATA,
                {"factor": "HYDRO_HAND", "method": f"HAND via D8 (channel acc>={CH_THRESH})",
                 "module": "HYDRO"})
    print("  HAND done")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    build(args.dem.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
