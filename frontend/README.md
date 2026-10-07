# Geo 前端（Vue 3 + TypeScript + Vite）

地理定位页面的独立前端（阶段一）。不修改后端路由，通过 Vite dev proxy 访问后端
`127.0.0.1:8000`。

## 命令

```bash
npm install
npm run dev        # http://127.0.0.1:5173
npm run typecheck  # vue-tsc 严格检查
npm run test       # vitest
npm run build      # vue-tsc + vite build
```

## API 契约

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| GET | `/api/geo/countries` | 国家知识库 |
| POST | `/api/geo/upload` | multipart 字段 `image` |
| POST | `/api/geo/{id}/feedback` | `confirm` / `reject` / `answer` / `free_text` |
| GET | `/api/geo/{id}/messages` | 会话消息 |
| POST | `/api/geo/{id}/analyze-point` | `{ x, y }`，取值 0..1 |

`/api`、`/media`、`/static` 已在 `vite.config.ts` 中代理到后端。

## 关于旧版页面

旧版 Jinja 页面（`templates/geo_index.html`）已随 legacy 源码移出仓库，完整副本保存在
本机 `.local-backup/legacy-2026-10-07/legacy-source/templates/`，不会上传或进入镜像。
当前新 Geo 前端不再链接该旧页面；login / game 等功能仍未迁移。

## 结构

- `src/api/`：请求封装与错误归一化（`ApiError` / `normalizeApiError`）。
- `src/types.ts`：接口返回类型。
- `src/composables/`：`useGeoSession` 会话状态机、`useMapViewport` 地图平移缩放。
- `src/components/`：上传、线索、地图、对话、确认弹窗、局部查看器等展示组件。
- `src/utils/`：线索分组去重、经纬度投影、评分格式化。
- `src/tests/`：API 错误、地图投影、线索分组、会话状态单测。

所有用户/后端文本都通过 Vue 插值渲染，未使用 `innerHTML` / `v-html`。

## 地图

底图 `public/world_map.png` 复制自 `static/images/world_map.png`，投影范围为
经度 -180..180、纬度 -60..80。

## 尚未实现（留待后续阶段）

- （已完成）Nginx / Docker 多阶段部署由根目录 `Dockerfile.frontend` + `deploy/nginx/default.conf` 提供。
- 与后端 SPA 路由的整合（例如 `/geo` 直接返回 SPA 的 `index.html`）。
- 图片上传进度、取消请求、键盘可达性增强。
