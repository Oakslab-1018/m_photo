# m-photo

照片地理定位实验项目。StreetCLIP 与 OCR 负责提供证据，用户对话补充线索，最终由同一个确定性评分器给出候选。

## 交互流程

上传图片 → StreetCLIP + OCR 融合 → 比较模型原始分差与融合分差：

- **高置信**：直接展示首选国家和依据，可继续细化城市。
- **中置信**：解释前三个候选，由用户选择确认。
- **低置信**：按候选差异提问，回答写入线索卡并重新评分；最多三问，仍拿不准就明确说明不确定。

补充线索入口位于引导面板底部，始终显示“补充线索，与定位助手对话”和“发送线索”，不必滚动到候选列表末尾。

模型预测不等于用户确认。只有显式确认才推进国家 → 城市；纠错和主动复核始终优先。LLM 仅抽取文字线索、组织提问和解释，不看图、不决定坐标；不可用时回退本地模板。

独立输入路名（如“玉林路”）也会记录为用户文字线索。已收录的街区关联可提出城市候选，未知路名仍保存在卡片中；路名不是唯一地点证明，最终城市需要确认。

分差不是概率；当前 `0.02/0.005` 门槛尚未校准。交互流程已通过相关回归；不能据此宣称对话提高了准确率。

## 评测状态

img2gps3k 已导入 2997 张，其中 2737 张有国家标签。旧基线由用户停止在 1764 张。557 张缓存的三臂回放已完成：纯模型、旧 OCR、新 OCR 的 Top-1 都是 49.2%，暂未证明准确率净收益。

全量三臂正在后台续跑，复用 598 条已完成记录，采集结束后自动离线对比。进度见 [ACTIVE_RUN.json](reports/ocr-evidence-ablation-20261020T205803/ACTIVE_RUN.json)，指标与复现见 [评测文档](data/eval/README.md)。真实参考索引和对话准确率 A/B 尚未完成。

## 启动

需要 Docker Desktop、Python 3.12 和 uv；本地 StreetCLIP 另需 CUDA GPU。

```powershell
Copy-Item .env.example .env  # 仅首次；已有 .env 不要覆盖
.\scripts\setup-streetclip.ps1
.\scripts\start-streetclip.ps1
docker compose up -d --build --wait
```

在 `.env` 中设 `STREETCLIP_ENABLED=true`，并为推理服务与后端配置同一个非空 `STREETCLIP_API_KEY`；无 GPU 时设为 `false`。访问 **http://127.0.0.1:8080**。修改环境配置后执行 `docker compose up -d --no-build --force-recreate --wait`。

## 边界与文档

- 图片和数据库位于 Docker 卷 `m-photo_backend-data`，不要执行 `docker compose down -v`。
- 默认仅本机访问，尚无用户鉴权。地图坐标是目录代表点，不是照片 GPS。
- [引导与置信路由](docs/guided-dialogue.md) · [三臂评测](docs/country-ablation.md) · [参考索引](docs/reference-index.md) · [目录](docs/geo-catalog.md) · [局部分析](docs/interaction-analysis.md) · [后端](backend/README.md)
