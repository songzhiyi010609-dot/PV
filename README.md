
# 商场光伏覆盖率 MVP

这个目录里是一个最小可行原型，用来验证这个统计流程：

```text
商场清单 + 商场卫星图裁剪
-> 判断每个商场是否疑似有光伏
-> 统计 有光伏商场数 / 商场总数
-> 输出表格、图片复核页、城市/省份汇总
```

当前版本故意不绑定大模型或深度学习框架，而是用 OpenCV 做一个可解释的“疑似光伏”初筛器。它的价值是先跑通业务闭环：数据格式、批量处理、判定阈值、人工复核、覆盖率报表。

## 快速运行

首次运行先创建并安装虚拟环境：

```powershell
cd C:\PV
python -m venv PV
.\PV\Scripts\python.exe -m pip install -r requirements.txt
```

之后运行 demo：

```powershell
cd C:\PV
.\PV\Scripts\python.exe scripts\run_demo.py
```

运行后查看：

- `outputs\mall_pv_results.csv`：逐商场判断结果
- `outputs\summary_by_city.csv`：城市覆盖率
- `outputs\summary_by_province.csv`：省份覆盖率
- `outputs\summary.md`：文字汇总
- `outputs\review.html`：人工复核页面，带原图和识别叠加图
- `outputs\overlays\`：每个商场的识别叠加图

## 换成真实商场数据

真实数据入口是 `data\malls.csv`。准备一个 CSV，至少包含这些列：

```csv
mall_id,name,province,city,lat,lon,image_path
mall_001,某某广场,江苏,苏州,31.2989,120.5853,data/images/mall_001.png
```

把商场屋顶或商场周边卫星图裁剪放到 `data\images\`，然后运行：

```powershell
.\PV\Scripts\python.exe scripts\run_mvp.py --malls data\malls.csv
```

如果你的图片路径是绝对路径，也可以直接填在 `image_path` 里。

### 只有真实图片，没有 CSV

把真实商场卫星图裁剪放到：

```text
C:\PV\data\real_malls\images
```

然后自动生成 `data\malls.csv`：

```powershell
.\PV\Scripts\python.exe scripts\create_real_malls_csv.py --image-dir data\real_malls\images --province 江苏 --city 苏州
.\PV\Scripts\python.exe scripts\run_mvp.py
```

如果你有一个元数据表，例如 `data\real_malls\malls_raw.csv`，其中有 `name/province/city/image_path` 或 `filename`，可以运行：

```powershell
.\PV\Scripts\python.exe scripts\create_real_malls_csv.py --metadata data\real_malls\malls_raw.csv --image-dir data\real_malls\images
.\PV\Scripts\python.exe scripts\run_mvp.py
```

### 用 H-RPVS 做真实影像测试

H-RPVS 是德国 Heilbronn 的屋顶光伏公开数据集，不是商场数据，但可以用来测试“真实高分辨率屋顶影像上的光伏识别”。

下载并解压 H-RPVS 到：

```text
C:\PV\data\H-RPVS
```

然后导入并运行：

```powershell
.\PV\Scripts\python.exe scripts\import_h_rpvs.py --source data\H-RPVS --limit 200 --run
```

## 当前判定口径

MVP 的默认口径是：

```text
若识别出的疑似光伏区域面积 >= 600 像素
且疑似光伏面积 / 图片面积 >= 0.15%
则判定该商场疑似有光伏
```

可以调阈值：

```powershell
.\PV\Scripts\python.exe scripts\run_mvp.py --malls data\malls.csv --min-pv-pixels 900 --min-coverage 0.002
```

## 下一步建议

1. 先选一个城市，收集 50-200 个商场样本。
2. 每个商场裁剪 300m-800m 范围卫星图，保证屋顶和停车棚可见。
3. 跑这个 MVP，打开 `outputs\review.html` 人工复核。
4. 把误判样本整理出来，后续换成 YOLO / SegFormer / DeepLabV3+。
5. 当人工复核准确率稳定后，再扩大到省级或全国。

## 重要说明

这个版本不是最终生产模型。它适合做原型验证和样本管理，不适合直接发布全国统计结论。真实项目里，光伏检测最好使用高分辨率影像和专门训练/微调的分割模型，并附带人工抽样复核误差。

Get-Item C:\PV\Git-RSCLIP\model.safetensors