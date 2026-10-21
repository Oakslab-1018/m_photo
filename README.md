# m-photo

照片地理定位实验项目。StreetCLIP 与 OCR 提供图像证据，用户对话补充线索，同一个确定性评分器给出国家候选。

## 交互流程

上传后，本地 OCR/语义分析先抽取线索，StreetCLIP 再对整图做国家排序；两份证据进入同一确定性 scorer，并按 `effective_margin = min(model_only_margin, fused_margin)` 路由：

| 档位 | 条件 | 行为 |
| --- | --- | --- |
| 高 | `effective_margin ≥ 0.02`，且模型与融合首位一致 | 展示 top1 与依据；用户显式确认后才进入城市 |
| 中 | 未达高，且 `effective_margin ≥ 0.005` 或融合改变首位 | 展示 top3，由用户选择确认 |
| 低 | 其余情况 | 按候选差异提问，最多 3 个国家线索问题；仍不确定则显式标记，或由用户从 top3 手动选择 |

`0.02/0.005` 是临时门槛，分差表示排序间距而非概率。纠错、冲突和主动复核优先于路由；城市阶段不继承国家档位。

- 只有显式确认才推进国家 → 城市。确认、排除和回退走确定性回复；城市图像排名复用缓存。
- 新线索优先复用图像排名并本地重排；核验城市缺少图像分时，按需定向评分并缓存，不伪造分数。补充线索入口在国家、城市和完成阶段均可使用。
- 局部报告中的真实 OCR 文字可点击“确认文字并用于定位”；后端从本会话报告读取原文和来源，确认文字不等于确认地点。
- 用户可直接输入路名、招牌或地标。系统保留原话，先查本地目录，再按需检索地点服务并核验所属城市；同名结果保留歧义并追问城市、区名或附近地标。只有唯一且可核验的城市证据参与评分，目录候选不伪造图像分数。
- 地名检索分别提示服务不可用、无结果或同名歧义；已有中文城市名也可离线精确查询。城市候选统一按有效融合分排序，文字来源不直接置顶。
- LLM 只做文字线索抽取、提问措辞和结果解释，不看图、不决定国家或坐标；联网地名查询由独立检索器执行，未接通用网页搜索。LLM 不可用时回退本地模板。
- 城市与确认路径优先显示已有中文名，其次用按 GeoNames ID 匹配的离线 Wikidata 标签，否则保留目录原名。
- 路由和交互改动目前只经用户手测，不能据此宣称对话提高了定位准确率。

## 评测状态

img2gps3k 全量三臂已完成：2997 张，2737 张有国家标签，0 分臂失败。Top-1：纯模型 **53.67%**、旧 OCR **52.83%**、新 OCR **53.45%**。新 OCR 优于旧规则，但未超过纯模型，暂无准确率净收益证据。

本轮复用 1355 条缓存；新增 1642 张的国家证据采集耗时约 23 分 35 秒，平均 0.86 秒/张。提速来自持久工作进程、CPU/GPU 流水线、StreetCLIP 批处理及独立 GPU OCR。该速度不含城市推理，不能直接与旧端到端 4 秒/张比较。

[完整报告](reports/eval-acceleration-20261021/run-20261021T121100/replay-retry-1/report.md) · [运行状态](reports/ocr-evidence-ablation-20261020T205803/ACTIVE_RUN.json) · [评测文档](data/eval/README.md)。真实参考索引与用户对话准确率 A/B 尚未完成。

## 启动

需要 Docker Desktop、Python 3.12 和 uv；本地 StreetCLIP 另需 CUDA GPU。

```powershell
Copy-Item .env.example .env  # 仅首次；已有 .env 不要覆盖
.\scripts\setup-streetclip.ps1
.\scripts\start-streetclip.ps1
docker compose up -d --build --wait
```

在 `.env` 中设 `STREETCLIP_ENABLED=true`，并为推理服务与后端配置同一个非空 `STREETCLIP_API_KEY`；无 GPU 时设为 `false`。访问 **http://127.0.0.1:8080**。修改环境配置后执行 `docker compose up -d --no-build --force-recreate --wait`。

独立 GPU OCR 可通过 `scripts/setup-gpuocr.ps1` 和 `scripts/start-gpuocr.ps1` 启动，依赖隔离于 `inference/gpuocr/.venv`。评测已使用 CUDA OCR；交互后端是否使用它由 `.env` 中的 `OCR_DEVICE=cpu/cuda` 控制。

## 边界与文档

- 图片和数据库位于 Docker 卷 `m-photo_backend-data`，不要执行 `docker compose down -v`。
- 默认仅本机访问，尚无用户鉴权。地图坐标是目录代表点，不是照片 GPS。
- [引导与置信路由](docs/guided-dialogue.md) · [三臂评测](docs/country-ablation.md) · [参考索引](docs/reference-index.md) · [目录](docs/geo-catalog.md) · [局部分析](docs/interaction-analysis.md) · [后端](backend/README.md)
