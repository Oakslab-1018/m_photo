# m-photo

照片地理定位实验项目：使用本地 StreetCLIP 检索候选，OCR 与引导对话补充线索，在本地国家/城市目录中逐步确认地点。DeepSeek 对话为可选文字辅助。

## 当前状态

- 可用：图片分析、国家 → 省级行政区 → 城市引导确认、线索否定/纠错/排除/回退、地图、局部分析和离线评测脚本。
- 参考索引的国家评分目前仅作诊断，不参与国家排序；纳入统一 scorer 仍属计划。
- 候选差异已有基础：`ModelAwareHypotheses` 会找 top-2 的不同特征维度并避免重复，但仍按固定维度优先级，暂无 margin/问题预算驱动的停止与弃权；同一 scorer 的有/无对话对比仍在计划中。
- LLM 当前仅有可选文字抽取/措辞，尚未接入 guided 建议问题；不接收图片、不猜坐标、不独立决定地点。
- 仓库没有真实评测图片或已标注 manifest，不能据此声称已有真实基线与准确率。

## 启动

需要 Docker Desktop、Python 3.12 和 uv；本地 StreetCLIP 另需 CUDA GPU。

```powershell
Copy-Item .env.example .env            # 仅首次；已有 .env 不要覆盖
.\scripts\setup-streetclip.ps1          # 首次安装权重
.\scripts\start-streetclip.ps1          # 启动本地推理
docker compose up -d --build --wait
```

启用本地模型时，在 `.env` 中设 `STREETCLIP_ENABLED=true`，并为本机推理服务与后端配置同一个非空 `STREETCLIP_API_KEY`（无需第三方申请）；无 GPU 时设 `STREETCLIP_ENABLED=false`。访问 **http://127.0.0.1:8080**；修改 `.env` 后执行：

```powershell
docker compose up -d --no-build --force-recreate --wait
```

## 评测入口

离线评测、最小 manifest、指标口径、严格退出行为和后续“当前 → 计划”安排统一见
[data/eval/README.md](data/eval/README.md)。评测脚本在宿主机运行；没有标注数据时只能生成骨架或做规则回退冒烟，不能产出模型准确率。

## 边界与文档

- 数据库和上传图片保存在 Docker 卷 `m-photo_backend-data`，不要执行 `docker compose down -v`。
- 默认仅限本机访问，尚无用户鉴权，不要直接暴露到公网。
- 相似度不是概率；国家和城市坐标是目录代表点，不是照片 GPS。
- 详细说明：[引导对话](docs/guided-dialogue.md) · [参考索引](docs/reference-index.md) · [目录与 GeoJSON](docs/geo-catalog.md) · [地图与局部分析](docs/interaction-analysis.md) · [后端](backend/README.md)
