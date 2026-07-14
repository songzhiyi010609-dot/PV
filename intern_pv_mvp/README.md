# 商场光伏潜力最小 MVP

这个小包是给实习生理解项目流程用的教学版 MVP。它不追求高准确率，也不依赖大模型或私有 API key，只保留项目最重要的两个步骤：

1. **确立 POI 并获取商场遥感图像**：读入商场清单，优先使用 CSV 里的经纬度；如果经纬度为空，则用 ArcGIS 免费地理编码做一个候选 POI；随后从 Esri World Imagery 下载商场周边卫星图。
2. **识别商场光伏和附近铺设条件**：用 OpenCV 颜色/形态规则粗略识别疑似光伏区域，并估算图中大面积规则屋顶候选，输出人工复核页。

> 重要：本 MVP 的识别逻辑是启发式规则，只适合教学、跑通流程、生成复核清单。不要把结果当成正式覆盖率或投资测算结论。

## 目录结构

```text
intern_pv_mvp/
  data/
    malls_sample.csv                    # 示例商场输入
  pv_mvp/
    poi.py                              # POI 解析：输入坐标或地理编码
    imagery.py                          # Esri 遥感瓦片下载与裁剪
    detect.py                           # 疑似光伏和屋顶候选规则识别
    report.py                           # summary.md 和 review.html
    io_utils.py                         # 通用路径/文件名工具
  scripts/
    01_resolve_poi_and_fetch_imagery.py # 第一步：POI + 遥感图
    02_detect_pv_and_potential.py       # 第二步：光伏 + 潜力
    run_all.py                          # 一键运行两步
  requirements.txt
  README.md
```

## 安装

建议新建虚拟环境：

```powershell
cd C:\PV\intern_pv_mvp
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

也可以直接使用已有 Python 环境，只要装好 `requirements.txt` 即可。

## 一键运行示例

```powershell
cd C:\PV\intern_pv_mvp
.\.venv\Scripts\python.exe scripts\run_all.py
```

运行完成后查看：

```text
outputs/
  poi_resolved.csv                    # POI 解析与遥感图路径
  imagery/                            # 下载的商场遥感图
  mall_pv_potential_results.csv       # 光伏和屋顶潜力初筛结果
  masks/                              # 疑似光伏 mask
  overlays/                           # 红色光伏叠加图、黄色屋顶候选图
  summary.md                          # 文字汇总
  review.html                         # 人工复核页
```

## 分步运行

第一步：确认 POI 并下载遥感图。

```powershell
.\.venv\Scripts\python.exe scripts\01_resolve_poi_and_fetch_imagery.py `
  --input data\malls_sample.csv `
  --output-dir outputs
```

第二步：识别疑似光伏和粗略铺设条件。

```powershell
.\.venv\Scripts\python.exe scripts\02_detect_pv_and_potential.py `
  --input outputs\poi_resolved.csv `
  --output-dir outputs
```

## 输入 CSV 格式

最小字段：

```csv
mall_id,name,city,address,lat,lon
36,上海南翔印象城MEGA,上海市,上海市嘉定区陈翔公路2299号,31.30695218,121.30116565
```

字段说明：

- `mall_id`：商场 ID。
- `name`：商场名称。
- `city`：城市，用于地理编码查询。
- `address`：地址，用于地理编码查询。
- `lat/lon`：推荐填写。若为空，脚本会用 `city + address + name` 调 ArcGIS 地理编码找候选点。

## 输出字段说明

`poi_resolved.csv`：

- `poi_status=ok`：拿到了可用中心点。
- `poi_source=input_coordinates`：直接使用 CSV 经纬度。
- `poi_source=arcgis`：通过 ArcGIS 地理编码得到候选点。
- `image_status=ok`：遥感图下载成功。
- `estimated_mpp`：当前 zoom 下的估算米/像素。

`mall_pv_potential_results.csv`：

- `pv_status`：`likely_pv / possible_pv / no_clear_pv`。
- `pv_ratio`：疑似光伏像素占比。
- `roof_candidate_ratio`：大面积规则屋顶候选像素占比。
- `potential_level`：`high / medium / low`，代表附近屋顶铺设条件的粗略初筛。
- `potential_reason`：给人工复核看的简短原因。

## 这个 MVP 和正式项目的区别

正式项目需要：

- 多源官方 POI 一致性验证，而不是只用一个坐标或一次地理编码。
- Git-RSCLIP/人工复核确认遥感图确实是目标商场主体。
- 更可靠的光伏分割模型，例如 BDAPPV、DeepPVMapper 或微调模型。
- 建筑轮廓、屋顶边界、遮挡、朝向、面积、容量密度等更完整的潜力测算。

这个 MVP 只负责让实习生先跑通闭环：**商场清单 -> POI 中心 -> 遥感图 -> 光伏初筛 -> 屋顶潜力初筛 -> 复核页**。
