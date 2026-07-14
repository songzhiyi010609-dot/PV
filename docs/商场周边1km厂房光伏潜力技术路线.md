# 商场周边 1km 厂房/仓库屋顶光伏潜力技术路线

## 项目定位

本项目定位为“商场周边 1km 厂房/仓库屋顶光伏潜力初筛与复核系统”，不是完全自动的投资级测算系统。

核心原则：

- 商场中心定位和光伏潜力评估解耦。
- 遥感图像只能判断“像不像商场/厂房/光伏”，不能单独确认具体 POI 身份。
- 正式评估只使用高置信商场中心；低置信中心必须人工复核。
- 结果要保留 `low/mid/high` 和不确定性，不输出无法解释的单点结论。

## 当前项目状态

- 项目目录：`C:\PV`
- 数据库：`C:\PV\malls_new.db`
- 上海商场样本：706 条
- 原始数据集：`C:\PV\datasets\shanghai_malls_satellite`
- Git-RSCLIP 模型：`C:\PV\Git-RSCLIP`
- 高置信商场中心：`C:\PV\outputs\experiments\20260708_relocate_mall_centers\data\mall_center_precise_pass_index.csv`
- 中/低置信复核清单：`C:\PV\outputs\experiments\20260708_relocate_mall_centers\data\mall_center_review_needed.csv`

当前最大风险不是光伏模型，而是商场中心点错位。一旦中心错，1km buffer、厂房识别、光伏统计都会跟着错。

## 推荐流水线

```text
商场数据库
  -> 商场名称/地址治理
  -> 官方 POI / 地址 / 地图候选点生成
  -> 多源坐标一致性 + 名称一致性 + Git-RSCLIP 视觉证据
  -> 商场中心 A/B/C/D 分级
  -> A/B 级中心生成 1km buffer
  -> buffer 影像切片
  -> 厂房/仓库候选筛选
  -> 已有光伏识别
  -> 可铺设面积与容量估算
  -> 人工复核闭环
  -> 实验记录和版本沉淀
```

## 商场中心分级

| 等级 | 定义 | 后续使用 |
| --- | --- | --- |
| A | 官方 POI 名称、地址、行政区、多源坐标、遥感形态均一致 | 可自动进入 1km 分析 |
| B | 多源位置接近，名称/图像有轻微不确定 | 可进入分析，但需抽检 |
| C | 候选冲突，地址/POI/图像证据不一致 | 必须人工复核 |
| D | 无法确认中心 | 暂停，不进入正式分析 |

当前 `mall_center_precise_pass_index.csv` 可以暂作为 A/B 候选输入，但后续应优先用官方 POI 验证脚本生成的 `auto_pass` 结果。

## 1km Buffer 数据结构

### mall_buffer

```text
run_id
mall_id
mall_name
center_lon
center_lat
center_source
center_confidence
buffer_radius_m
tile_count
imagery_source
imagery_zoom
created_at
```

### image_tile

```text
run_id
tile_id
mall_id
mall_name
center_lon
center_lat
tile_center_lon
tile_center_lat
offset_x_m
offset_y_m
distance_to_center_m
buffer_radius_m
tile_size_px
overlap_px
estimated_mpp
tile_ground_width_m
image_status
image_path
imagery_source
imagery_zoom
```

## 厂房/仓库屋顶识别方向

优先路线：

1. 如果有建筑轮廓，先用 building footprint 作为屋顶对象。
2. 如果没有建筑轮廓，先用 1km buffer 切片做候选筛选。
3. 使用 Git-RSCLIP 对 tile 或 roof crop 做语义筛查：
   - positive: factory roof, warehouse roof, logistics warehouse, industrial buildings
   - negative: residential buildings, road, park, school, hospital, shopping mall
4. 使用形态规则过滤：
   - 面积大
   - 矩形规则
   - 成组出现
   - 周边硬化地面/园区道路/物流堆场
5. 低置信样本进入人工复核。

## 光伏识别方向

已有 BDAPPV / DeepPVMapper 可作为光伏初筛模型。

必须增加后处理：

- 光伏 mask 必须和屋顶候选高度重叠。
- 过滤蓝色彩钢瓦、玻璃屋顶、水体、阴影、停车棚误检。
- 对高潜力样本和疑似误检样本生成人工复核图。

## 潜力估算口径

先做初筛级估算：

```text
usable_area_mid = roof_area_m2 * usable_ratio_mid - existing_pv_area_m2
capacity_mid_kwp = usable_area_mid * capacity_density_mid
annual_generation_mid_kwh = capacity_mid_kwp * local_specific_yield
```

建议默认参数：

- 大型平屋顶厂房 usable_ratio_mid: 0.65
- 彩钢瓦坡屋顶 usable_ratio_mid: 0.55
- 复杂屋顶 usable_ratio_mid: 0.35
- capacity_density_mid: 0.15 kWp/m2

正式报告必须输出 low/mid/high 三档和 `uncertainty_level`。

## 必须人工复核的情况

- 商场中心 C/D 级。
- 同名商场多个 POI。
- 视觉像商场但 POI 名称弱匹配。
- 1km buffer 影像质量差。
- 厂房/仓库候选低置信。
- 光伏疑似误检。
- 预测潜力排名靠前的高价值样本。

## 当前优先落地步骤

1. 运行官方 POI 身份验证，生成 A/B/C/D 或 `auto_pass/manual_review/reject_candidate`。
2. 用高置信中心生成 1km buffer tile 索引。
3. 小样本下载 1km 切片，人工观察切片尺度是否合适。
4. 使用 Git-RSCLIP 对 tile 做厂房/仓库语义筛查。
5. 对高分 factory/warehouse tile 跑光伏识别。
6. 输出 mall-level 初版潜力汇总。
