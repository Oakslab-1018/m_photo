# m-photo · 第一阶段：AI 地理定位（Geo）新系统

本仓库正在从旧版 Flask 应用迁移到一个模块化新系统。**当前 `compose.yaml`
只启动第一阶段的新 Geo 系统**，由两个服务组成：

| 服务 | 容器内 | 宿主机 | 说明 |
| --- | --- | --- | --- |
| `frontend` | Nginx `:80` | 默认 `127.0.0.1:8080`（可覆盖） | 托管 Vue 3 构建产物（`dist/`），并把 `/api/`、`/media/` 反向代理到 `backend:8000` |
| `backend` | FastAPI `:8000` | 仅内部网络（`expose`，不发布端口） | `uvicorn backend.app.main:app`，使用 `uv.lock` 锁定的 Python 3.12 依赖 |

数据全部落在具名卷 **`backend-data`**（挂载到容器内 `/data`），包含 SQLite
数据库与上传图片。Compose 内部固定 `M_PHOTO_DATA_DIR=/data`、
`GEO_AI_MODE=mock`，因此无需配置任何密钥即可启动。

> **安全提示**：第一阶段 geo 系统**没有用户鉴权**，因此 Compose 默认只把前端绑定到
> 回环地址 `127.0.0.1`，**不是开箱即用的公网多用户安全版本**。不要直接暴露到公网；
> 如需局域网访问，请用 `BIND_ADDRESS` / `APP_PORT` 显式覆盖，并自行承担鉴权、
> TLS、访问控制等安全责任。

> **旧 Flask 系统的位置**：旧应用源码已移出仓库根目录，完整快照保存在本机
> `.local-backup/legacy-2026-10-07/`（`legacy-source/`），该目录已加入
> `.gitignore` / `.dockerignore`，不会上传或进入镜像。登录流程（login）、猜图游戏（game）
> 与旧数据自动迁移仍**未迁移到新 Geo 系统**，本文档不承诺任何自动迁移。
> 旧 `database.db` 与 static 图片已一并备份在备份目录的 `user-data/` 下，不会随镜像发布。

---

## 快速开始

前置条件：Docker Engine / Docker Desktop 已启动，并带有 `docker compose` v2。
先用 `docker info` 确认 daemon 可连通，再进行构建部署。

```powershell
# 在仓库根目录执行；首次构建会下载 Python/Node/Nginx 基础镜像与依赖，耗时较长
docker compose up -d --build
```

启动后访问：

- 前端：<http://127.0.0.1:8080>
- 后端健康检查（经由 Nginx 代理）：<http://127.0.0.1:8080/api/health>

> 后端 8000 端口只在 Compose 内部网络暴露（`expose: 8000`），不映射到宿主机；
> 如需本机直连后端，请用 `docker compose exec backend ...` 或临时覆盖 compose 配置。
> 前端默认只监听回环地址，仅同一台机器可访问。

## 查看状态与日志

```powershell
docker compose ps
docker compose logs -f backend
docker compose logs -f frontend

# 单次探测健康状态（镜像未安装 curl，用后端自带的 Python 标准库探测）
docker compose exec backend python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).read().decode())"
```

`frontend` 使用 `depends_on: backend.condition: service_healthy`：只有在 backend
健康检查（`GET /api/health`）通过后才会启动。两个服务都配置
`restart: unless-stopped`，容器异常退出或宿主机重启后会自行恢复。

## 数据卷、备份与恢复

具名卷 `backend-data` 是**唯一**的持久化位置：

```powershell
docker volume ls --filter name=m-photo
docker volume inspect m-photo_backend-data
```

备份时**先停 backend**，让 SQLite 处于一致状态（frontend 可以暂时保持运行）：

```powershell
# 1) 停 backend
docker compose stop backend

# 2) 只读挂载具名卷，把完整数据打包到当前目录
docker run --rm -v m-photo_backend-data:/data:ro -v "${PWD}:/backup" alpine:3.20 tar czf /backup/backend-data-backup.tar.gz -C /data .

# 3) 备份完成后恢复服务
docker compose start backend
```

跨平台（bash / macOS / Linux）等价命令：

```bash
docker compose stop backend
docker run --rm -v m-photo_backend-data:/data:ro -v "$PWD:/backup" alpine:3.20 tar czf /backup/backend-data-backup.tar.gz -C /data .
docker compose start backend
```

> 如果确实无法安排停服窗口，不要直接复制正在写入的 `m_photo.db`；应对 SQLite 使用
> 在线备份（例如 `sqlite3 /data/m_photo.db ".backup '/backup/m_photo.db'"`），并单独
> 备份上传目录，再自行校验一致性。

恢复采用**非破坏性**流程：把备份还原到一个新的具名卷，原卷保持不动，人工确认后再切换。
以下步骤**不会清空或覆盖现有数据**：

```powershell
# 1) 停服务（保留现有卷 backend-data）
docker compose down

# 2) 新建一个空目标卷
docker volume create m-photo_restore

# 3) 把备份解压到新卷（此时新卷为空，不触碰旧数据）
docker run --rm -v m-photo_restore:/data -v "${PWD}:/backup" alpine:3.20 tar xzf /backup/backend-data-backup.tar.gz -C /data

# 4) 新建临时文件 compose.restore.yaml（内容如下），把 backend 指向新卷后启动核对
# services:
#   backend:
#     volumes:
#       - m-photo_restore:/data
# volumes:
#   m-photo_restore:
#     name: m-photo_restore   # 显式卷名：与 docker volume create 的名字一致，避免 Compose 加项目前缀
docker compose -f compose.yaml -f compose.restore.yaml up -d
```

确认新卷中的数据库与上传图片无误、且已有可用备份之后，再**由人工决定**是否清理旧卷。

> 若要**永久**切换到恢复卷，请保留该 override 文件，或把 `compose.yaml` 中
> backend 的卷映射改为 `m-photo_restore:/data`（并在顶层 volumes 声明
> `name: m-photo_restore`）。不要在 Compose 仍引用 `backend-data` 时删除旧卷后直接
> `up`：Compose 会把缺失的具名卷重新创建为空卷，等同于数据丢失。

删除具名卷是不可恢复操作，必须停服并二次确认：

```powershell
# 仅在人工确认后才执行；docker compose down 保留卷，真正删除数据的是 docker volume rm
docker compose down
docker volume rm m-photo_backend-data
```

> `docker compose down` **默认保留**具名卷；只有显式执行
> `docker compose down -v` 才会删除数据卷（会一并删除数据库与上传图片，慎用）。

## 环境变量

日常使用无需 `.env`。示例见 [`.env.example`](.env.example)，只列出少量变量；
Compose 已内置默认值。**不要提交真实密钥**：第一阶段 Geo 系统不依赖任何 API Key，
也无需在镜像或仓库中硬编码密钥。

| 变量 | Compose 默认 | 用途 |
| --- | --- | --- |
| `BIND_ADDRESS` | `127.0.0.1` | 前端发布地址；改为 `0.0.0.0` 才允许局域网访问（当前无鉴权，谨慎） |
| `APP_PORT` | `8080` | 前端宿主机端口 |
| `M_PHOTO_DATA_DIR` | 容器内固定 `/data` | 可写数据根目录（数据库 + 上传图片），由 `compose.yaml` 写入，不由 `.env` 覆盖 |
| `GEO_AI_MODE` | 容器内固定 `mock` | `mock` 为无模型启发式分析；`real` 暂未打包 |

修改绑定地址或端口有两种方式：写入 `.env`（从 `.env.example` 复制，勿提交真实密钥），
或临时覆盖（PowerShell）：

```powershell
$env:BIND_ADDRESS = "0.0.0.0"
$env:APP_PORT = "8081"
docker compose up -d
```

## 本地开发（不使用 Docker）

后端（Python 3.12 + uv）：

```powershell
uv sync
uv run uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

前端（Vite 开发服务器，`vite.config.ts` 已将 `/api`、`/media` 代理到
`127.0.0.1:8000`）：

```powershell
Set-Location frontend
npm ci
npm run dev        # http://127.0.0.1:5173
```

其他常用命令：

```powershell
Set-Location frontend
npm run build      # 生产构建，等价于 Dockerfile.frontend 的构建阶段
Set-Location ..
uv run pytest      # 后端测试（开发依赖，不进入生产镜像）
```

## mock 与 real 模式

- **mock（默认，Compose 固定）**：使用基于图像颜色/启发式的模拟分析，不下载任何
  模型权重，镜像更小、启动即可用。Docker 镜像只支持 mock。
- **real**：调用 easyocr 等真实模型，首次运行需下载模型权重。**当前 Docker 镜像
  未打包 OCR 模型与相关依赖**，因此 `GEO_AI_MODE=real` 在容器内不受支持。
  若要使用 real 模式，请在本地环境按后端文档安装额外依赖与权重。

## 构建说明与限制

实际的镜像定义如下，避免与 Dockerfile 内容产生歧义：

- `Dockerfile.backend`：两个构建阶段。阶段一是
  `FROM ghcr.io/astral-sh/uv:0.12.23 AS uv`，仅提供 `uv`/`uvx` 二进制；
  阶段二是 `FROM python:3.12-slim-bookworm AS runtime`，通过
  `COPY --from=uv /uv /uvx /bin/` 取得 uv，然后执行
  `uv sync --frozen --no-dev --no-install-project`，严格按 `pyproject.toml` +
  `uv.lock` 安装运行依赖。容器以非 root（UID 10001）运行，不安装 OCR 额外依赖。
  第一阶段不再单独复制根目录 `ai/` 或 `reasoning/`；`COPY backend/ ./backend/` 已包含
  迁移后的领域实现 `backend/app/domain/geolocation/`（推理与知识图谱）及
  `backend/app/domain/geolocation/analysis/`（分析流水线与分析器）。
  启动命令为 `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`。
- `Dockerfile.frontend`：两个构建阶段。构建阶段为
  `FROM node:24-bookworm-slim AS build`，执行 `npm ci` + `npm run build`；
  运行阶段为 `FROM nginx:1.27-alpine AS runtime`，只复制 `dist/` 与 Nginx 配置，
  **不使用 Vite 生产服务器**。
- 健康检查使用 Python 标准库 `urllib`，无需安装 `curl`；后端镜像不执行
  `apt-get` 安装额外系统包。
- `.dockerignore` / `.gitignore` 排除了 Windows 下的 `.venv`、`.local-backup/`、
  `backend/.venv`、`backend/tests`、`node_modules`、`.env`、`storage/`、上传目录、
  日志、`database.db` 与本地模型权重，避免它们被烘进镜像或误提交。

## 故障排查

```powershell
# 1) 确认 Compose 配置可解析（不需要 daemon）
docker compose config

# 2) 查看服务状态与健康检查详情
docker compose ps
docker inspect --format "{{json .State.Health}}" $(docker compose ps -q backend)

# 3) 若 frontend 一直未启动：backend 未通过健康检查，先看 backend 日志
docker compose logs --tail=200 backend
```

常见原因：Docker daemon 未启动、首次构建下载慢、`frontend/package-lock.json`
缺失导致 `npm ci` 失败、宿主机 8080 端口被占用（设置 `APP_PORT` 覆盖）。

---

## 范围与后续

**第一阶段范围**：仅 Geo 新系统的容器化（FastAPI + Vue/Nginx 双服务）。
**不在本阶段**：旧 Flask 的 login / game 功能、旧数据自动迁移、real OCR 模型打包。
**安全边界**：无用户鉴权、默认仅回环绑定，不是公网多用户安全版。

后续步骤（尚未实现，不作承诺）：旧功能逐项迁移、旧数据导出/导入工具、real 模式
可选镜像与模型权重管理。