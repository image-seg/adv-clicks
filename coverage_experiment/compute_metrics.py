"""Compute click coverage and Spearman correlations from supplied IoU values.

Standard library only; no model inference or external experiment folders.
Run: python compute_metrics.py
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parent
STRATEGIES = ("MIN", "BASE", "MAX")
EPS = 1e-9


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_iou(row):
    value = float(row["iou"])
    require(math.isfinite(value) and 0 <= value <= 1, "IoU must be finite and within [0,1]")
    return value


def read_inputs():
    clicks = read_csv(ROOT / "clicks_1st_round.csv")
    images = sorted({r["image_id"] for r in clicks})
    require(images == [f"image_{i:03d}" for i in range(1, 41)],
            "Expected exactly image_001 through image_040")
    click_groups = defaultdict(list)
    click_lookup = {}
    image_shapes = {}
    for row in clicks:
        image_id, click_id = row["image_id"], row["click_id"]
        require(click_id not in click_lookup, "Duplicate click ID")
        require(int(row["is_positive"]) == 1, "Unexpected click sign")
        cw, ch = int(row["coordinate_width"]), int(row["coordinate_height"])
        width, height = int(row["image_width"]), int(row["image_height"])
        x, y = int(row["x"]), int(row["y"])
        shape = cw, ch, width, height
        require(all(v > 0 for v in shape) and 0 <= x < cw and 0 <= y < ch,
                "Invalid image size or evaluation coordinates")
        require(image_shapes.setdefault(image_id, shape) == shape,
                "Inconsistent dimensions for the same image")
        for axis, coordinate, size, native in (("x", x, cw, width), ("y", y, ch, height)):
            require(abs(float(row["image_" + axis]) - ((coordinate + 0.5) * native / size - 0.5)) < 1e-8,
                    "Native-image coordinate mapping mismatch")
        click_groups[image_id].append(click_id)
        click_lookup[click_id] = image_id
    require(len(clicks) == 600, "Expected 600 clicks")
    for image_id in images:
        require(len(click_groups[image_id]) == 15, "Expected 15 clicks per object")
        click_groups[image_id].sort()
    for folder, extension in (("images", ".jpg"), ("masks", ".png")):
        require(sorted(p.stem for p in (ROOT / folder).glob("*" + extension)) == images,
                "Expected one image and one mask for each object")

    records = read_csv(ROOT / "metrics.csv")
    require(len(records) == 41400 and all(r["source"] in ("user", "attack") for r in records),
            "Expected 41400 user/attack metric rows")
    user_rows = [r for r in records if r["source"] == "user"]
    models = sorted({r["model_id"] for r in user_rows})
    require(len(models) == 23, "Expected exactly 23 models")
    users, attacks = {}, {}
    for row in records:
        require(row["model_id"] in models and row["image_id"] in images,
                "Metric row has an unknown model or image")
        if row["source"] == "user":
            key = row["model_id"], row["image_id"], row["click_id"]
            require(key not in users and click_lookup.get(key[2]) == key[1]
                    and row["strategy"] == "" and int(row["click_number"]) == 1,
                    "Invalid/duplicate user metric row")
            users[key] = read_iou(row)
        else:
            key = row["model_id"], row["image_id"], row["strategy"], int(row["click_number"])
            require(key not in attacks and row["click_id"] == "" and key[2] in STRATEGIES
                    and 1 <= key[3] <= 10, "Invalid/duplicate attack metric row")
            attacks[key] = read_iou(row)
    require(len(users) == len(models) * len(clicks) == 13800, "Incomplete user metrics")
    require(len(attacks) == len(models) * len(images) * len(STRATEGIES) * 10 == 27600,
            "Incomplete attack metrics")
    return models, images, click_groups, users, attacks


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            result[index] = (start + 1 + end) / 2
        start = end
    return result


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = fmean(rx), fmean(ry)
    dx, dy = [v - mx for v in rx], [v - my for v in ry]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denominator if denominator else None


def text_table(headers, rows, right_aligned=()):
    """Render an aligned plain-text table without external dependencies."""
    cells = [list(map(str, row)) for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]

    def line(row):
        return " | ".join(value.rjust(widths[i]) if i in right_aligned else value.ljust(widths[i])
                          for i, value in enumerate(row)).rstrip()

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([line(cells[0]), separator, *(line(row) for row in cells[1:])])


def text_report(model_statistics, correlations, total_covered, total_results):
    ordered = sorted(model_statistics, key=lambda r: (-r["covered"] / r["total"], r["model_id"]))
    coverage_rows = [(i, r["model_id"], f"{r['covered']}/{r['total']}",
                      f"{100 * r['covered'] / r['total']:.2f}") for i, r in enumerate(ordered, 1)]
    coverage_rows.append(("", f"Mean ({len(ordered)} models)", f"{total_covered}/{total_results}",
                          f"{100 * total_covered / total_results:.2f}"))
    coverage_table = text_table(["#", "Model", "Covered / total", "Coverage (%)"],
                                coverage_rows, right_aligned=(0, 2, 3))
    correlation_table = text_table(["Comparison", "Models", "Spearman rho"],
                                   correlations, right_aligned=(1, 2))
    return ("Coverage (IoU, first click; ordered MIN/MAX)\n"
            "40 images; 600 user clicks; 23 models\n\n" + coverage_table +
            "\n\nSpearman correlations (IoU)\n\n" + correlation_table + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    models, images, clicks, users, attacks = read_inputs()
    model_statistics = []
    for model in models:
        stats = []
        covered = 0
        for image_id in images:
            lo, hi = sorted((attacks[model, image_id, "MIN", 1], attacks[model, image_id, "MAX", 1]))
            values = [users[model, image_id, c] for c in clicks[image_id]]
            inside = sum(lo - EPS <= q <= hi + EPS for q in values)
            below = sum(q < lo - EPS for q in values)
            above = sum(q > hi + EPS for q in values)
            require(inside + below + above == 15, "Coverage partition failed")
            covered += inside
            stats.append(dict(attack_min=lo, attack_D=hi - lo,
                              user_min=min(values), user_D=max(values) - min(values)))
        total = sum(len(clicks[image_id]) for image_id in images)
        summary = {key: fmean(r[key] for r in stats) for key in stats[0]}
        summary.update(model_id=model, covered=covered, total=total)
        model_statistics.append(summary)

    total_covered = sum(r["covered"] for r in model_statistics)
    total_results = sum(r["total"] for r in model_statistics)
    require(total_results == len(users) and abs(total_covered / total_results - fmean(
        r["covered"] / r["total"] for r in model_statistics)) < 1e-12,
        "Mean model coverage and total coverage disagree")
    correlations = []
    for comparison, x_key, y_key in (("AttackMIN_vs_UserMIN", "attack_min", "user_min"),
                                      ("AttackD_vs_UserD", "attack_D", "user_D")):
        rho = spearman([r[x_key] for r in model_statistics], [r[y_key] for r in model_statistics])
        correlations.append((comparison, len(models), rho))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = text_report(model_statistics, correlations, total_covered, total_results)
    (out / "results.txt").write_text(report, encoding="utf-8", newline="\n")
    print(report, end="")


if __name__ == "__main__":
    main()
