# m-photo

照片地理定位实验项目：使用本地 StreetCLIP 提供候选证据，通过引导式对话逐步确认国家、省级行政区和城市。支持线索核实、地点排除、追问、未知回答和回退；DeepSeek 对话为可选辅助。

## 主要功能

- 上传图片，在源目录（250 个国家/地区）内获取相似度排名；应用将台湾、香港、澳门地区候选归入中国，并保留地区来源和城市信息。
- 新会话在同一个引导框内逐条确认照片中的观察，可返回上一条修改；全部处理后点击“下一步”查看国家 / 地区候选可信度前五，再明确确认进入省级行政区 → 城市，各层级均可返回上一步。否定线索与排除地点分别处理，支持恢复排除。
- 选定国家后在该国城市目录（`data/geo_catalog.sqlite3`，34156 个城市）内检索城市候选；引导流程按已确认行政区筛选完整目录，并支持名称检索。
- 国家、省级行政区、城市统一保存在 `data/geo_catalog.sqlite3`，支持 SQLite 索引查询及按范围分页的 GeoJSON 点输出；后端首次启动将快照导入运行库，后续启动不再解析目录 JSON。
- 多维线索卡涵盖文字、地名、文化、行驶方向及道路、地形、水体等观察，支持独立滚轮滚动；对话补充的观察会动态添加到线索卡并刷新候选前五，可靠地名证据参与国家和城市候选排序。
- 对话工作区支持补充、否定和纠正线索，发送后显示处理状态、理解结果和候选变化，失败可重试。
- 地图随国家 → 省级行政区 → 城市的候选选择、确认和回退动态聚焦，坐标缺失时只回退到真实父级代表点。
- 局部分析支持点选或框选、调整范围与文字/场景侧重，返回文字框、识别证据、画质提示、场景候选和核实建议。
- 可配置 PP-OCRv5 server 完整检测、文字方向分类和识别模型；真实 OCR 安装与验证见 `docs/ocr.md`。
- 会话、图片和完整模型排名持久化，重启后可继续。
- 模型不可用时明确提示，并只依据可靠文字、用户补充等线索回落推断；颜色产生的通用场景观察仅作展示，避免固定国家排名。对话接口不可用时回落到模板。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 前端 | Vue 3、TypeScript、Vite |
| 后端 | FastAPI、SQLAlchemy、SQLite、Alembic |
| 本地推理 | StreetCLIP、PyTorch、Transformers，独立 Python 环境 |
| 部署 | Docker Compose、Nginx；GPU 推理在宿主机运行 |

## 快速启动

需要 Docker Desktop、Python 3.12 和 uv；本地 StreetCLIP 另需支持 CUDA 的 NVIDIA GPU。以下命令在 Windows PowerShell 的项目根目录执行。

首次使用先复制配置，**已有 `.env` 不要覆盖**：

```powershell
Copy-Item .env.example .env
```

启用本地模型时修改 `.env`：

```dotenv
STREETCLIP_ENABLED=true
STREETCLIP_URL=http://host.docker.internal:8001
STREETCLIP_API_KEY=replace-with-random-token
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-flash
```

`STREETCLIP_API_KEY` 是后端与本机推理服务共用的口令，不需要申请第三方 Key。DeepSeek Key 留空即可使用模板对话。

```powershell
.\scripts\setup-streetclip.ps1   # 首次安装并下载权重；后续复用缓存
.\scripts\start-streetclip.ps1   # 后台启动本地 GPU 推理

docker compose up -d --build --wait
```

访问 **http://127.0.0.1:8080**。不使用 GPU 时保持 `STREETCLIP_ENABLED=false`，跳过模型安装和启动即可。

> 依赖按清华 TUNA 镜像源锁定（`uv.lock` 记录下载地址，`--frozen` 不再解析）：本网络直连
> `files.pythonhosted.org` 会被限速到约 118KB/s 并出现 `tls handshake eof`，镜像构建取
> 58MiB 的 `opencv-python-headless` 反复失败。换回官方源需重新锁定并同步改
> `Dockerfile.backend` 的 `UV_DEFAULT_INDEX`：`uv lock --default-index https://pypi.org/simple`。

修改 `.env` 后重新加载容器配置：

```powershell
docker compose up -d --no-build --force-recreate --wait
```

## 数据与使用边界

- 数据库和上传图片保存在 Docker 卷 `m-photo_backend-data`；**不要运行 `docker compose down -v`**。
- 模型权重缓存：`inference/streetclip/storage/huggingface/`。Windows 重启后需重新运行模型启动脚本。
- `.env`、权重、运行数据和备份不提交到 Git。
- 默认仅限本机访问，尚无用户鉴权，不要直接暴露到公网。
- 候选目录来自 `data/geo_catalog.sqlite3`（250 个国家/地区）：余弦相似度不是概率，地图标记是国家代表点，不是照片 GPS。
- 城市候选同样来自只读目录 `data/geo_catalog.sqlite3`，后端逐行核对 id、名称与坐标；城市坐标也是代表点。
  城市行的 `region` 是 GeoNames 的一级行政区**代码**（如东京为 `40`），不是地区名称。
- 带标注的参考街景索引是可选的：未配置时后端不下发相关字段；配置后命中结果只作检索诊断展示，不参与排序。
- StreetCLIP 按非商业许可使用。DeepSeek 仅接收脱敏文字上下文，不发送图片；可辅助理解用户文字，提取结果经校验后参与推理，接口不可用时使用本地解析和模板。道路、地形、水体等缺少目录特征的观察会标明仅供参考，不单独改变排序。

详细说明：[后端文档](backend/README.md) · [引导对话协议](docs/guided-dialogue.md) · [目录入库与 GeoJSON](docs/geo-catalog.md) · [地图与局部分析](docs/interaction-analysis.md) · [模型接入说明](docs/streetclip-deployment-plan.md)
