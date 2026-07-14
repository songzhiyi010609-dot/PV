from __future__ import annotations

from html import escape
import os
from pathlib import Path

import pandas as pd


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            mall_count=("mall_id", "count"),
            suspected_pv_count=("has_pv", "sum"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    grouped["coverage_rate"] = grouped["suspected_pv_count"] / grouped["mall_count"]
    grouped["coverage_rate_pct"] = (grouped["coverage_rate"] * 100).round(2)
    grouped["avg_confidence"] = grouped["avg_confidence"].round(3)
    return grouped.sort_values(["coverage_rate", "mall_count"], ascending=[False, False])


def _rel(path: str, base: Path) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return p.relative_to(base).as_posix()
    except ValueError:
        return Path(os.path.relpath(p, base)).as_posix()


def _write_html(df: pd.DataFrame, output_dir: Path) -> None:
    cards = []
    for _, row in df.sort_values(["has_pv", "confidence"], ascending=[False, False]).iterrows():
        image_src = escape(_rel(str(row.get("resolved_image_path", "")), output_dir))
        overlay_src = escape(_rel(str(row.get("overlay_path", "")), output_dir))
        badge = "suspected PV" if bool(row.get("has_pv")) else "not detected"
        badge_class = "yes" if bool(row.get("has_pv")) else "no"
        coverage_pct = float(row.get("coverage", 0.0)) * 100

        cards.append(
            f"""
            <article class="card">
              <header>
                <h2>{escape(str(row.get("name", "")))}</h2>
                <span class="badge {badge_class}">{badge}</span>
              </header>
              <p class="meta">{escape(str(row.get("province", "")))} / {escape(str(row.get("city", "")))} · confidence {float(row.get("confidence", 0.0)):.2f} · coverage {coverage_pct:.2f}%</p>
              <div class="imgs">
                <figure>
                  <img src="{image_src}" alt="source image">
                  <figcaption>source</figcaption>
                </figure>
                <figure>
                  <img src="{overlay_src}" alt="overlay image">
                  <figcaption>overlay</figcaption>
                </figure>
              </div>
              <p class="reason">{escape(str(row.get("reason", "")))}</p>
            </article>
            """
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>商场光伏 MVP 复核页</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f6f7f8; color: #1f2933; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    .lead {{ margin: 0 0 24px; color: #52606d; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
    .card {{ background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06); }}
    header {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    h2 {{ font-size: 16px; margin: 0; }}
    .badge {{ font-size: 12px; color: #fff; border-radius: 999px; padding: 4px 8px; white-space: nowrap; }}
    .yes {{ background: #0f766e; }}
    .no {{ background: #64748b; }}
    .meta {{ font-size: 13px; color: #52606d; margin: 8px 0 12px; }}
    .imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border: 1px solid #d9e2ec; border-radius: 6px; background: #e5e7eb; }}
    figcaption {{ font-size: 12px; color: #616e7c; margin-top: 4px; }}
    .reason {{ font-size: 12px; color: #3e4c59; margin: 10px 0 0; line-height: 1.45; }}
    @media (max-width: 520px) {{
      main {{ padding: 14px; }}
      .grid {{ grid-template-columns: 1fr; }}
      .imgs {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>商场光伏 MVP 复核页</h1>
    <p class="lead">红色叠加区域是 MVP 初筛出的疑似光伏。这个页面用于人工复核，不代表最终生产模型结果。</p>
    <section class="grid">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""
    (output_dir / "review.html").write_text(html, encoding="utf-8")


def write_reports(df: pd.DataFrame, *, output_dir: str | Path, project_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = int(len(df))
    suspected = int(df["has_pv"].sum()) if total else 0
    rate = suspected / total if total else 0.0

    by_city = _summary(df, ["province", "city"])
    by_province = _summary(df, ["province"])
    by_city.to_csv(output_dir / "summary_by_city.csv", index=False, encoding="utf-8-sig")
    by_province.to_csv(output_dir / "summary_by_province.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 商场光伏覆盖率 MVP 汇总",
        "",
        f"- 商场总数：{total}",
        f"- 疑似有光伏商场数：{suspected}",
        f"- 覆盖率：{rate:.2%}",
        "",
        "## 按省份",
        "",
        by_province.to_markdown(index=False),
        "",
        "## 按城市",
        "",
        by_city.to_markdown(index=False),
        "",
        "说明：当前结果来自 OpenCV 启发式初筛，需要人工复核。",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8-sig")
    _write_html(df, output_dir)
