# m-photo

照片地理定位实验项目。StreetCLIP 与 OCR 提供图像证据，用户对话补充线索，同一个确定性评分器给出国家与城市候选。

## 交互流程

流程：上传 → 分析 → 核对观察 → 确认国家 → 确认城市 → 结果

1. 上传照片后，本地 OCR 抽取文字线索，StreetCLIP 对整图做国家排序；两份证据进入同一个确定性 scorer。
2. 先核对需要复核的观察，再展示国家候选。高/中档通常跳过逐条复核；纠错、冲突和主动复核仍会阻断。
3. 只有用户显式确认才推进：国家 → 城市 → 完成。排序、自由文本和 LLM 都不会自动确认。
4. 国家、城市和完成阶段都可补充文字线索，或确认报告中的 OCR 原文；新线索重新进入 scorer，但不直接确认地点。
5. 可回退、恢复已排除项、重新核对观察。城市位置是目录代表点，不是照片 GPS。

| 档位 | 条件 | 行为 |
| --- | --- | --- |
| 高 | `effective_margin ≥ 0.02` 且模型与融合首位一致 | 只展示 top1 |
| 中 | `effective_margin ≥ 0.005`，或融合改变首位 | 展示 top3，由用户选择确认 |
| 低 | 其余情况 | 最多 3 个国家线索问题；之后显式标记不确定，或从 top3 手动选择 |

`effective_margin = min(model_only_margin, fused_margin)`。`0.02/0.005` 是临时门槛，表示排序间距而非概率，尚未校准；城市阶段不继承国家档位，纠错、冲突和主动复核优先于路由。

文字线索先查本地城市名与道路/POI 索引，必要时调用地点服务，并可配置网页搜索兜底；网页结果只召回目录城市，补齐图像分后统一排序，不直接增加地点匹配分。LLM 只做经校验的线索抽取、提问措辞和带来源解释，不决定国家或坐标。城市场景观察可参与有界重排，属实验信号。详见[引导对话](docs/guided-dialogue.md)、[地点证据](docs/place-evidence.md)、[城市线索](docs/city-clues.md)。

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

在 `.env` 中设 `STREETCLIP_ENABLED=true`，并为推理服务与后端配置同一个非空 `STREETCLIP_API_KEY`；无 GPU 时设为 `false`。访问 <http://127.0.0.1:8080>。修改环境配置后执行 `docker compose up -d --no-build --force-recreate --wait`。

独立 GPU OCR 可通过 `scripts/setup-gpuocr.ps1` 和 `scripts/start-gpuocr.ps1` 启动，依赖隔离于 `inference/gpuocr/.venv`。评测已使用 CUDA OCR；交互后端是否使用它由 `.env` 中的 `OCR_DEVICE=cpu/cuda` 控制。

## 边界与文档

- 图片和数据库位于 Docker 卷 `m-photo_backend-data`，不要执行 `docker compose down -v`。
- 默认仅本机访问，尚无用户鉴权。地图坐标是目录代表点，不是照片 GPS。
- [引导与置信路由](docs/guided-dialogue.md) · [三臂评测](docs/country-ablation.md) · [参考索引](docs/reference-index.md) · [目录](docs/geo-catalog.md) · [局部分析](docs/interaction-analysis.md) · [后端](backend/README.md)
