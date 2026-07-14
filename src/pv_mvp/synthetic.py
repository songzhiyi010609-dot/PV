from __future__ import annotations

from pathlib import Path
from random import Random

import cv2
import numpy as np
import pandas as pd


def _draw_roof(img: np.ndarray, rng: Random, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (95, 100, 105), 2)
    for _ in range(rng.randint(3, 8)):
        vx = x + rng.randint(12, max(12, w - 12))
        vy = y + rng.randint(12, max(12, h - 12))
        cv2.rectangle(img, (vx - 5, vy - 5), (vx + 5, vy + 5), (135, 138, 140), -1)


def _draw_pv_array(img: np.ndarray, x: int, y: int, rows: int, cols: int, panel_w: int = 34, panel_h: int = 11) -> None:
    for r in range(rows):
        for c in range(cols):
            px = x + c * (panel_w + 5)
            py = y + r * (panel_h + 5)
            cv2.rectangle(img, (px, py), (px + panel_w, py + panel_h), (42, 58, 78), -1)
            cv2.rectangle(img, (px, py), (px + panel_w, py + panel_h), (18, 31, 48), 1)
            cv2.line(img, (px + panel_w // 2, py + 1), (px + panel_w // 2, py + panel_h - 1), (64, 78, 96), 1)


def _base_scene(seed: int) -> np.ndarray:
    rng = Random(seed)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:, :] = (162, 174, 146)

    # roads and surrounding parcels
    cv2.rectangle(img, (0, 220), (512, 270), (116, 120, 119), -1)
    cv2.rectangle(img, (230, 0), (282, 512), (124, 126, 124), -1)
    cv2.rectangle(img, (15, 15), (205, 205), (142, 164, 123), -1)
    cv2.rectangle(img, (305, 304), (500, 495), (126, 156, 118), -1)

    roof_color = rng.choice([(178, 181, 180), (166, 170, 172), (154, 158, 163)])
    _draw_roof(img, rng, 92, 92, 328, 278, roof_color)
    cv2.rectangle(img, (146, 140), (356, 322), (188, 193, 193), -1)
    cv2.rectangle(img, (146, 140), (356, 322), (112, 118, 122), 2)

    noise = np.random.default_rng(seed).normal(0, 4, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_demo_dataset(project_dir: str | Path) -> Path:
    project_dir = Path(project_dir)
    image_dir = project_dir / "data" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ("mall_001", "示例商场A", "上海", "上海", 31.2304, 121.4737, True, "large rooftop PV"),
        ("mall_002", "示例商场B", "江苏", "苏州", 31.2989, 120.5853, False, "plain roof"),
        ("mall_003", "示例商场C", "广东", "深圳", 22.5431, 114.0579, True, "parking canopy PV"),
        ("mall_004", "示例商场D", "四川", "成都", 30.5728, 104.0668, False, "blue skylight distractor"),
        ("mall_005", "示例商场E", "浙江", "杭州", 30.2741, 120.1551, True, "small rooftop PV"),
    ]

    rows = []
    for idx, (mall_id, name, province, city, lat, lon, has_pv, note) in enumerate(samples, start=1):
        img = _base_scene(100 + idx)
        if has_pv:
            if idx == 3:
                cv2.rectangle(img, (72, 388), (440, 438), (137, 137, 132), -1)
                _draw_pv_array(img, 90, 400, 2, 8, panel_w=33, panel_h=10)
            elif idx == 5:
                _draw_pv_array(img, 182, 180, 3, 4, panel_w=30, panel_h=10)
            else:
                _draw_pv_array(img, 170, 172, 6, 5, panel_w=32, panel_h=11)
        else:
            if idx == 4:
                cv2.rectangle(img, (190, 170), (322, 252), (104, 145, 168), -1)
                cv2.rectangle(img, (190, 170), (322, 252), (132, 170, 190), 2)

        image_path = image_dir / f"{mall_id}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append(
            {
                "mall_id": mall_id,
                "name": name,
                "province": province,
                "city": city,
                "lat": lat,
                "lon": lon,
                "image_path": f"data/images/{mall_id}.png",
                "demo_label_has_pv": has_pv,
                "note": note,
            }
        )

    malls_csv = project_dir / "data" / "malls_sample.csv"
    pd.DataFrame(rows).to_csv(malls_csv, index=False, encoding="utf-8-sig")
    return malls_csv
