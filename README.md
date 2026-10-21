# m-photo

照片地理定位实验项目。StreetCLIP 与 OCR 负责提供证据，用户对话补充线索，最终由同一个确定性评分器给出候选。

## 交互流程

上传图片 → StreetCLIP + OCR 融合 → 比较模型原始分差与融合分差：

- **高置信**：直接展示首选国家和依据，可继续细化城市。
- **中置信**：解释前三个候选，由用户选择确认。
- **低置信**：按候选差异提问，回答写入线索卡并重新评分；最多三问，仍拿不准就明确说明不确定。

补充线索入口位于引导面板底部，始终显示“补充线索，与定位助手对话”和“发送线索”，不必滚动到候选列表末尾。

城市候选、确认路径和地图标记使用按 GeoNames ID 匹配的离线中文名；缺少可靠译名时显示“中文名待补充”。普通确认、排除和回退使用确定性回复，城市图像排名复用缓存，新增线索只触发本地重排。

模型预测不等于用户确认。只有显式确认才推进国家 → 城市；纠错和主动复核始终优先。LLM 仅抽取文字线索、组织提问和解释，不看图、不决定坐标；不可用时回退本地模板。

用户可直接输入路名、招牌或地标。系统保存原话，先查本地城市目录，再按需检索地点服务并核验所属城市；同名地点保留歧义并追问城市、区名或附近地标。只有可核验且唯一的城市证据才参与原有候选评分，目录候选不伪造图像分数。

分差不是概率；当前 `0.02/0.005` 门槛尚未校准。本轮交互修改由用户手测；不能据此宣称对话提高了准确率。

## 评测状态

img2gps3k 全量三臂已完成：2997 张、2737 张有国家标签、0 分臂失败。Top-1：纯模型 **53.67%**、旧 OCR **52.83%**、新 OCR **53.45%**。新 OCR 优于旧规则，但未超过纯模型，尚无准确率净收益证据。

本轮复用 1355 条缓存；新增 1642 张的国家证据采集耗时约 23 分 35 秒，平均 0.86 秒/张。采用持久工作进程、CPU/GPU 流水线、StreetCLIP 批处理及独立 GPU OCR。此速度不包含城市推理，不能直接与旧端到端 4 秒/张比较。

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
