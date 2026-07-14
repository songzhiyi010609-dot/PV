# 商场中心人工复核系统

这是一个内网可访问的本地 Web 复核系统，用于确认商场中心点是否可信。系统不会修改原始数据库或原始遥感数据，只在 `C:/PV/outputs/review/mall_center_review` 下写入复核数据库、队列、卫星图缓存和导出 CSV。

## 配置文件

主配置：

```text
C:/PV/review_system/config/review_config.json
```

本地账号密码：

```text
C:/PV/review_system/config/users.local.json
```

如果 `users.local.json` 不存在，系统启动时会自动从 `users.local.example.json` 生成。第一次启动后请打开它，把 `replace_with_password` 改成真实密码，否则系统会拒绝登录。

## 安装依赖

```powershell
cd C:\PV
C:\PV\PV\Scripts\python.exe -m pip install -r C:\PV\review_system\requirements-review.txt
```

## 启动

```powershell
cd C:\PV
C:\PV\PV\Scripts\python.exe C:\PV\review_system\app.py --host 0.0.0.0 --port 8787
```

本机访问：

```text
http://127.0.0.1:8787
```

同事内网访问：

```text
http://你的电脑内网IP:8787
```

如果 Windows 防火墙提示，允许 Python 在专用网络访问。

## 复核等级

- `A`：高置信通过，可进入 1km 分析。
- `B`：人工确认可用，但建议抽查，可进入 1km 分析。
- `C`：继续待复核，不能进入 1km 分析。
- `D`：拒绝/错误/无法定位，不能进入 1km 分析。

## 输出

```text
C:/PV/outputs/review/mall_center_review/review.db
C:/PV/outputs/review/mall_center_review/review_queue.csv
C:/PV/outputs/review/mall_center_review/review_results.csv
C:/PV/outputs/review/mall_center_review/mall_center_review_approved.csv
C:/PV/outputs/review/mall_center_review/review_summary.md
C:/PV/outputs/review/mall_center_review/satellite_crops
```

`mall_center_review_approved.csv` 是后续 1km 厂房/光伏潜力分析优先使用的中心点表。
