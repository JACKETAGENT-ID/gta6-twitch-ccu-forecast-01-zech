"""gta6-twitch-ccu-forecast-01 entry (zech): per-hour log-linear regression on comparables.

For each forecast hour h = 2..24 the program fits, at runtime and from the
comparables shipped inside the case file, an ordinary least squares model

    log(viewers_h) = a_h + b_h * log(p1) + c_h * log(channels_1)

where p1 is the case's hour-1 print and channels_1 its hour-1 channel count.
No coefficient is hardcoded: every fit is computed from the case's own
comparables when the program runs, so the same code adapts to any case.

The fit is local in scale: a comparable is weighted by how close its own
hour-1 print is to this case's, with a Gaussian kernel of bandwidth 2 in
log units, and comparables more than a factor of 10 away in hour-1 print
are left out entirely. Games of a similar size share a launch-day shape;
tiny launches (whose later hours explode as regions unlock) would otherwise
drag the fit of a huge launch around. When too few comparables survive the
window, the fit falls back to plain least squares over all of them.

The median forecast is exp(fitted value). The quantile spread comes from the
fit's residuals at that hour: sigma_h is the residual standard deviation
(floored so the band is never overconfident), and the five quantiles are
median * exp(z_tau * sigma_h) with z the standard normal quantiles
(-1.645, -0.674, 0, 0.674, 1.645).

Fallbacks keep every case valid:
  - if the case has no usable hour-1 channel count, or too few comparables
    carry channels, the model drops to one predictor, log(p1) only;
  - if even that fit is impossible, the median is p1 times the median
    comparable ratio viewers_h / p1, with the spread from the log ratios.

Python 3.13, standard library only. No network, no randomness, no clock:
two runs on the same input write byte-identical output. Runtime is far
under the 8 second limit (23 tiny least-squares fits over ~40 comparables).
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA_FLOOR = 0.15
SIGMA_FLOOR_FALLBACK = 0.25
SIGMA_CAP = 1.2
MIN_ROWS_TWO_PRED = 8


def _parse_ts(text):
    """ISO 8601 timestamp (with Z) -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination.

    Returns the solution list, or None when the matrix is (near-)singular.
    """
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col] / piv
                for k in range(col, n + 1):
                    aug[r][k] -= factor * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def wls(rows, nfeat, weights):
    """Weighted least squares with intercept (weights from scale closeness).

    rows: list of (xs, y); weights: same length, non-negative. Returns
    (beta, residuals) or None when the normal equations are singular.
    """
    p = nfeat + 1
    normal = [[0.0] * p for _ in range(p)]
    rhs = [0.0] * p
    for (xs, y), w in zip(rows, weights):
        if w == 0.0:
            continue
        vec = [1.0] + list(xs)
        for i in range(p):
            rhs[i] += w * vec[i] * y
            for j in range(p):
                normal[i][j] += w * vec[i] * vec[j]
    beta = solve_linear(normal, rhs)
    if beta is None:
        return None
    residuals = []
    for xs, y in rows:
        vec = [1.0] + list(xs)
        fitted = sum(beta[i] * vec[i] for i in range(p))
        residuals.append(y - fitted)
    return beta, residuals


def extract_comparables(case):
    """Release-day curves of the case's comparables, aligned to hour 1.

    Each comparable contributes its hour-1 print (p1), its hour-1 channel
    count (c1, None when missing) and its viewers for hours 2..24. A
    comparable without an hour-1 row (e.g. the GTA VI category's own
    pre-release history) contributes nothing and is skipped.
    """
    out = []
    for comp in sorted(case.get("comparables", []), key=lambda c: c.get("name", "")):
        day_start = comp.get("day_start")
        hourly = comp.get("hourly") or []
        if not day_start or not hourly:
            continue
        try:
            t0 = _parse_ts(day_start)
        except ValueError:
            continue
        by_hour = {}
        for row in hourly:
            try:
                dt = _parse_ts(row["from"])
            except (KeyError, ValueError):
                continue
            k = int((dt - t0).total_seconds() // 3600)
            if 0 <= k < 24:
                v = row.get("viewers")
                if isinstance(v, (int, float)) and v > 0:
                    by_hour[k] = row
        if 0 not in by_hour:
            continue
        p1 = float(by_hour[0]["viewers"])
        c1_raw = by_hour[0].get("channels")
        c1 = float(c1_raw) if isinstance(c1_raw, (int, float)) and c1_raw > 0 else None
        series = {h: float(by_hour[h]["viewers"]) for h in range(1, 24) if h in by_hour}
        if len(series) < 12:
            continue
        out.append({"p1": p1, "c1": c1, "series": series})
    return out


def fit_hour(comps, hour, p1, c1):
    """Median and log-spread for one forecast hour.

    Fits log(viewers_hour) on log(p1) and log(channels_1) across the
    comparables that have that hour, weighted toward comparables whose own
    hour-1 print is close to this case's, then evaluates the fit at this
    case's own hour-1 print and channel count. Returns (median, sigma).
    """
    rows_two = []
    rows_one = []
    for comp in comps:
        viewers = comp["series"].get(hour - 1)
        if viewers is None:
            continue
        y = math.log(viewers)
        x1 = math.log(max(1.0, comp["p1"]))
        rows_one.append(((x1,), y))
        if comp["c1"] is not None:
            rows_two.append(((x1, math.log(max(1.0, comp["c1"]))), y))
    use_two = c1 is not None and len(rows_two) >= MIN_ROWS_TWO_PRED
    rows = rows_two if use_two else rows_one
    nfeat = 2 if use_two else 1
    # Local weights: Gaussian kernel (bandwidth 2 in log units) on the
    # comparable's hour-1 print vs this case's, cut off beyond one dex.
    log_p1 = math.log(max(1.0, p1))
    weights = []
    for (xs, _y) in rows:
        d = abs(xs[0] - log_p1)
        weights.append(math.exp(-((d) / 2.0) ** 2) if d <= math.log(10) else 0.0)
    if sum(1 for w in weights if w > 0.0) < nfeat + 3:
        weights = [1.0] * len(rows)
    if len(rows) >= nfeat + 3:
        fitted = wls(rows, nfeat, weights)
    else:
        fitted = None
    if fitted is not None:
        beta, residuals = fitted
        xs = [log_p1]
        if use_two:
            xs.append(math.log(max(1.0, c1)))
        mu = beta[0] + sum(beta[i + 1] * xs[i] for i in range(nfeat))
        dof = max(1, len(residuals) - nfeat - 1)
        sw = sum(weights)
        sigma = math.sqrt(
            sum(w * r * r for w, r in zip(weights, residuals))
            / max(1.0, sw * (dof / len(residuals)))
        )
        return math.exp(mu), min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
    # Fallback: median comparable ratio viewers_h / p1, spread from log ratios.
    ratios = [
        math.log(comp["series"][hour - 1] / max(1.0, comp["p1"]))
        for comp in comps
        if hour - 1 in comp["series"]
    ]
    if not ratios:
        return max(1.0, p1), 0.5
    ratios.sort()
    median_ratio = ratios[len(ratios) // 2]
    mean_ratio = sum(ratios) / len(ratios)
    if len(ratios) > 1:
        sigma = math.sqrt(sum((r - mean_ratio) ** 2 for r in ratios) / (len(ratios) - 1))
    else:
        sigma = 0.5
    return math.exp(math.log(max(1.0, p1)) + median_ratio), min(
        SIGMA_CAP, max(SIGMA_FLOOR_FALLBACK, sigma)
    )


def round2(x):
    """x rounded to 2 decimals, half up (same as the kit starter)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    c1_raw = first.get("channels")
    c1 = float(c1_raw) if isinstance(c1_raw, (int, float)) and c1_raw > 0 else None
    comps = extract_comparables(case)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        median, sigma = fit_hour(comps, h, p1, c1)
        qs = []
        for z in Z:
            q = median * math.exp(z * sigma)
            q = min(1e9, max(0.01, q))
            qs.append(round2(q))
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
