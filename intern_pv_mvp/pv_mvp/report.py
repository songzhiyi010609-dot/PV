from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, relative_to_base


def write_summary(results: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    total = int(len(results))
    processed = int((results["process_status"] == "ok").sum()) if total and "process_status" in results else 0
    likely = int((results["pv_status"] == "likely_pv").sum()) if total and "pv_status" in results else 0
    possible = int((results["pv_status"] == "possible_pv").sum()) if total and "pv_status" in results else 0
    high = int((results["potential_level"] == "high").sum()) if total and "potential_level" in results else 0
    medium = int((results["potential_level"] == "medium").sum()) if total and "potential_level" in results else 0

    lines = [
        "# MVP 运行汇总",
        "",
        "本结果来自轻量规则模型，只用于理解流程和生成复核清单，不用于正式统计。",
        "",
        f"- 商场数：{total}",
        f"- 成功识别：{processed}",
        f"- likely_pv：{likely}",
        f"- possible_pv：{possible}",
        f"- high potential：{high}",
        f"- medium potential：{medium}",
        "",
        "## 字段解释",
        "",
        "- `pv_status`：商场图中疑似已有光伏状态。",
        "- `roof_candidate_ratio`：图中大面积规则屋顶候选占比，粗略代表附近可铺设条件。",
        "- `potential_level`：基于屋顶候选占比和已有光伏占比的初筛等级。",
        "- `review.html`：人工复核入口，红色为疑似光伏，黄色为屋顶候选。",
    ]
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_review_html(results: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    cards = []
    for _, row in results.iterrows():
        if row.get("process_status") != "ok":
            cards.append(
                f"""
                <article class="card">
                  <h2>{escape(str(row.get("mall_id", "")))} · {escape(str(row.get("name", "")))}</h2>
                  <p class="meta">Status: <b>{escape(str(row.get("process_status", "")))}</b></p>
                  <p class="reason">{escape(str(row.get("process_error", "not processed")))}</p>
                </article>
                """
            )
            continue
        image_path = escape(relative_to_base(row["image_path"], output_dir))
        pv_overlay = escape(relative_to_base(row["pv_overlay_path"], output_dir))
        roof_overlay = escape(relative_to_base(row["roof_overlay_path"], output_dir))
        cards.append(
            f"""
            <article class="card">
              <h2>{escape(str(row["mall_id"]))} · {escape(str(row["name"]))}</h2>
              <p class="meta">PV: <b>{escape(str(row["pv_status"]))}</b> · Potential: <b>{escape(str(row["potential_level"]))}</b> · roof ratio {float(row["roof_candidate_ratio"]) * 100:.2f}%</p>
              <div class="images">
                <figure><img src="{image_path}"><figcaption>source satellite crop</figcaption></figure>
                <figure><img src="{pv_overlay}"><figcaption>red = suspected PV</figcaption></figure>
                <figure><img src="{roof_overlay}"><figcaption>yellow = roof candidates</figcaption></figure>
              </div>
              <p class="reason">{escape(str(row["potential_reason"]))}</p>
            </article>
            """
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>商场光伏潜力 MVP 复核页</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f6f7f9; color: #1f2937; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 24px; margin: 0 0 6px; }}
    .lead {{ margin: 0 0 20px; color: #5b6776; }}
    .card {{ background: #fff; border: 1px solid #d9dee8; border-radius: 8px; padding: 14px; margin-bottom: 16px; }}
    h2 {{ font-size: 17px; margin: 0 0 8px; }}
    .meta, .reason {{ color: #52606d; font-size: 13px; }}
    .images {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border: 1px solid #d9dee8; border-radius: 6px; }}
    figcaption {{ font-size: 12px; color: #687386; margin-top: 4px; }}
    @media (max-width: 760px) {{ .images {{ grid-template-columns: 1fr; }} main {{ padding: 14px; }} }}
  </style>
</head>
<body>
  <main>
    <h1>商场光伏潜力 MVP 复核页</h1>
    <p class="lead">这个页面用于教学和人工复核：先看 POI 取图是否正确，再看红色光伏和黄色屋顶候选是否合理。</p>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    path = output_dir / "review.html"
    path.write_text(html, encoding="utf-8")
    return path
