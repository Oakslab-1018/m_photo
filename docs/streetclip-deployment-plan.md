# StreetCLIP 本地推理与 DeepSeek 会话接入

## 当前结构

```text
Vue / Nginx :8080 -> FastAPI（Docker，内部 :8000）
                       |-> 本机 StreetCLIP :8001 -> RTX GPU
                       |-> DeepSeek Chat Completions（仅在设置 Key 后）
```

StreetCLIP 环境在 `inference/streetclip/.venv`，与业务环境隔离，但共用本机显卡。
业务镜像没有 Torch/Transformers；这里只安装 HTTP 客户端。
推理服务只启动一个模型实例、单次一张图，预计算并复用全目录文本 embedding。

## 已选依赖与权重

- Python 3.12；Torch 2.9.1 + cu128；Transformers 4.57.6。
- 模型 `geolocal/StreetCLIP`，固定 revision `e3561ba2ad9bf14c9efd6b0092607b8497efbfea`。
- `.bin` 权重及处理器/分词器均缓存到 `inference/streetclip/storage/huggingface`。
- GPU 环境、缓存、日志、PID 和密钥不提交，也不进入 Docker 构建上下文。
- 模型许可 CC-BY-NC-4.0，按非商业使用；这不是完整 PIGEON 复现。

## 安装与启动（仓库根目录，PowerShell）

```powershell
# 已安装过时会复用环境和下载缓存。首次安装需要下载较大的 CUDA wheel 和权重。
.\scripts\setup-streetclip.ps1
.\scripts\start-streetclip.ps1
Invoke-RestMethod http://127.0.0.1:8001/health
```

后台启动无可见窗口，PID 和 stdout/stderr 在 `inference/streetclip/storage/`。
停止用 `.\scripts\stop-streetclip.ps1`；前台排错用 `start-streetclip.ps1 -Foreground`。
Windows 重启后需重新启动模型进程。Docker 后端不会自动替你启动本机 Python 进程。

本机 Docker Desktop 已验证 `host.docker.internal:8001` 可以访问回环绑定的服务，
因此继续保持模型 `127.0.0.1` 绑定，不需改成局域网公开端口。
只有明确需要其他绑定方式时使用 `-BindHost`；非回环绑定必须配置 bearer token。

根目录 `.env` 用于 Compose 和本机模型进程。后端常用配置：

```dotenv
STREETCLIP_ENABLED=true
STREETCLIP_URL=http://host.docker.internal:8001
STREETCLIP_API_KEY=<本机推理服务与后端使用同一个随机 token>
STREETCLIP_TIMEOUT_SECONDS=45
STREETCLIP_DEVICE=cuda
STREETCLIP_CITY_LIMIT_PER_COUNTRY=100
STREETCLIP_REFERENCE_INDEX=
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_TIMEOUT_SECONDS=15
```

已在本机生成模型服务 token，不需要申请第三方 StreetCLIP API Key。
新增环境可从 `.env.example` 复制并自行生成随机 token；不要提交真实 `.env`。
原生后端开发时将 `STREETCLIP_URL` 改为 `http://127.0.0.1:8001`，
用 `uv run --frozen uvicorn backend.app.main:app --env-file .env` 加载配置。
`GEO_AI_MODE=mock` 只控制旧 OCR/启发式分支，不会关闭已启用的 StreetCLIP。

## DeepSeek：以后只填 Key

默认 API 模型 ID 是 `deepseek-flash`（DeepSeek V4.1 Flash），允许通过环境变量修改。
Key 留空时不发出 API 请求；模型排序与模板对话仍可使用。
填写 `.env` 的 `DEEPSEEK_API_KEY` 后执行：

```powershell
docker compose up -d --no-build --force-recreate --wait
```

重建容器会重新读取环境；同时重建 Nginx 避免后端地址变化导致旧 DNS 缓存。
不需要重新下载 StreetCLIP、不需要把 Key 写进前端，也不要使用 `docker compose down -v`。
`/api/health` 的 `deepseek_configured` 仅表示 Key 非空，不代表 Key/余额已通过实际验证。
没有用户 Key，因此真实 DeepSeek 请求仍未验收；已验证缺 Key、超时、401、5xx 和坏响应的兜底。

DeepSeek 只接收经过白名单和长度限制的结构化线索、候选与最近对话；原始 OCR/证据文字不发送，
对话中的网址、邮箱、常见令牌及配置密钥会脱敏。不会发送照片字节、
本地图片路径或 EXIF/GPS。填写 Key 即意味着允许这些文字发送到第三方。
它负责解释和提问，不直接修改国家排序、排除集合或确认状态。
自由文本线索抽取仍保留原关键词解析，不宣称已实现任意自然语言的结构化地理推理。

## 排名、证据与事务

`POST /predict` 接收 multipart 字段 `file` 和 bearer token，返回全部目录国家（当前 250 个）的 `raw_cosine`、
模型 revision、候选/模板 hash、Top1-Top2 margin 和 `uncertain`。
每个文本模板先 L2 归一化，模板平均后再 L2；图像 embedding 单独归一化后做余弦匹配。
上传进入模型前会处理 EXIF 方向、转 RGB、重编码并去掉元数据。

Alembic `0002_model_evidence` 增加可空 `model_evidence_json`，原会话和图片不变。
完整排序单独持久化，不伪造成 `Clue(dimension="country")`。
排除反馈会过滤并重排完整先验，不再次调用 GPU；局部点击分析保留全图先验。
只对明确的用户补充线索加有限规则排序权重，颜色/亮度启发式不会压过模型。
网络请求发生在写事务之前；对话先在内存预演，写入时动作不一致就丢弃生成文字。
确认操作保持确定性，不调用对话模型。

前端显示名次、余弦相似度或规则权重，不再把这些分数标成百分比。
候选范围是完整国家目录；目录外照片仍会得到目录内 Top1。
低 margin 只是排序不确定，不是可靠的目录外检测。地图坐标是国家代表点，不是照片定位坐标。
两个客户端都没有自动重试；HTTP socket timeout 不承诺严格的总墙钟时间。
模型服务故障时回落到原规则；对话模型故障时回落到模板。

### 城市检索与参考街景

选中国家后前端调用 `POST /api/geo/{session_id}/cities`，后端用该会话原图调用
`POST /predict-cities`（multipart `file` + 表单 `country_iso`），返回该国目录内的城市候选。
城市由 `data/geo_catalog.sqlite3` 中的城市目录定义，不由模型定义：后端逐行核对 id、名称与经纬度，
并校验 `scope.city_catalog_revision` 等于库内保存的城市逻辑 revision，对不上就整份丢弃。
`city_id` 在 API 层序列化为十进制字符串（前端以它为 key）。
城市候选同样是代表点，不是照片定位；`region` 是 GeoNames admin1 **代码**（例如东京为 `40`），不是地区名。
该接口只读且无状态：不写消息、不改国家排序、不影响已确认国家，切换国家查看城市不会污染分析结果。

带标注的参考街景索引（`STREETCLIP_REFERENCE_INDEX`）是可选的。加载成功后 `/predict` 会附带
`reference` sidecar；后端校验 revision 后，把命中结果作为**只读诊断**挂在对应国家候选上。
参考余弦不进入排序公式，因此不会改变候选顺序；sidecar 损坏只丢诊断，国家排名照常返回。
未配置索引时这三个字段整个不出现，前端据此区分"没有库"和"有库但没命中"。

## 本机验证记录

- RTX 5060 Laptop GPU；CUDA 可用，实际服务 `/health` 为
  `ready=true, device=cuda, countries=250, city_capability=true, cities_loaded=34156,
  reference_configured=false`，其 `city_catalog_revision` 与后端 `data/geo_catalog.sqlite3` 中的城市目录 保存的城市逻辑 revision 一致。
- 仓库公开 `nagasaki.jpg` 示例：日本排第一，原始余弦约 0.225（`hybrid_rank`，含 OCR 线索加成后 0.2631）；
  首次请求约 9.65 秒（含模型预热），随后完整请求约 1.0 秒。
  这只是单张示例实测，不是准确率或普遍延迟承诺。
- 城市链路实测（同一会话）：`POST /api/geo/{id}/cities` 对 JP 返回 `status=ok`、
  `cities_considered=100`、10 行候选，约 1.05 秒；`city_id` 为十进制字符串，坐标与名称逐行对上目录。
  TK 返回 `no_cities`（目录里没有该国的城市），`ZZ` 返回 422，未知会话返回 404，小写 `jp` 正常归一化。
  连续 5 次城市调用后会话消息数不变，确认该接口不写库。
- 局部点击实测：新增 4 条线索、裁剪区域 281px，存储的 evidence 多出 `crops` 而全图 250 国先验保持不变，
  且 `crops` 条目里没有 `reference` 键（sidecar 只描述全图）。
- 参考街景 sidecar 由推理服务与后端两侧的测试覆盖；仓库内没有真实标注语料
  （`data/reference-manifest.example.json` 是空的），因此未做真实语料端到端验收。
- 独立 Compose 数据卷验收：上传 -> 排除 JP 后 KR 首位 -> 局部点击 -> 重启保留完整先验和图片 -> 确认成功。
- 自动化基线：后端 199 tests + 75 subtests；推理服务 46 tests（另 1 skipped）；
  前端 42 tests、类型检查和构建通过。

来源：StreetCLIP 官方 Hugging Face 模型卡；PyTorch 官方 previous-versions；
DeepSeek 官方 pricing / Chat Completions 文档。运行结果详见忽略目录下的验收 JSON。
