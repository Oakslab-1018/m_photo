# 交互式地图（MapLibre GL JS）

## 配置

地图样式通过 Vite 的 public build-time 变量注入，全部会内联进前端产物，
**因此都不是服务端密钥**：

| 变量 | 说明 |
| --- | --- |
| `VITE_MAP_STYLE_URL` | 完整 MapLibre style JSON URL，优先级最高 |
| `VITE_MAPTILER_API_KEY` | MapTiler public key；自动解析为 `https://api.maptiler.com/maps/streets-v4/style.json?key=…` |

- 本地如需覆盖默认底图：复制 `frontend/.env.example` 为 `frontend/.env.local` 后填写。
- Docker Compose：在根目录 `.env` 中设置同名变量；Compose 会将两者转发为前端
  build args（默认值均为空）。变量在构建期内联，修改后需重建前端镜像：
  ```bash
  docker compose build frontend && docker compose up -d frontend
  ```
- MapTiler hybrid 样式提供卫星影像和标注；可用显式 style URL 覆盖（将
  `<public-key>` 替换为受限的公开 key）：
  `VITE_MAP_STYLE_URL=https://api.maptiler.com/maps/hybrid/style.json?key=<public-key>`
- 两项都为空时，默认加载随前端构建发布的 Esri World Imagery raster style
  （`frontend/public/esri-world-imagery-style.json`），无需 API key。该样式固定使用
  `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}`、
  256px 瓦片和 0–23 级缩放，并在 MapLibre attribution control 中显示
  “Source: Esri, Vantor, Earthstar Geographics, and the GIS User Community”。
- `VITE_MAP_STYLE_URL` 优先于 `VITE_MAPTILER_API_KEY`；MapTiler 仅在显式提供 key
  时启用。
- MapTiler key 是公开的客户端 key，必须在 MapTiler 控制台配置允许的来源
  （origins/referrers）限制；不要在 `VITE_*` 中放任何服务端密钥。

## 沉浸式推演工作区

地图始终挂载并铺满工作区，照片证据与推演任务以悬浮面板呈现，不再与地图上下争抢高度。

- 桌面端：左侧照片与证据面板宽 312px，右侧推演面板宽 432px，可展开至 560px；
  两侧均可收起。顶部显示国家 → 城市确认路径、会话状态和历史入口。
- “协同推演”同时显示证据与任务；“地图专注”让出地图画布；“照片取证”优先显示
  证据台。原图与区域分析沿用现有查看器，切换面板不会提交确认或丢弃输入草稿。
- 小屏端：证据台默认收起；手机使用摘要、半屏、全屏三档底部任务面板，提供明确按钮。
  照片、线索和参考资料使用单独的证据浮层。
- 推演面板保留当前阶段和路径，正文单独滚动，补充线索输入固定在底部。
  完整推理记录从顶部打开独立抽屉，仍遵守现有核对阶段的显示门槛。
- 候选卡编号与地图标记一致，已确认地点显示勾选标记。选中仅触发高亮和聚焦；
  地点确认、排除、撤回排除和返回上一级仍使用原有服务端流程。
- 线索只在附有有效 `extra.bbox_norm` 或 `extra.region_bbox_norm` 时显示“查看原图位置”。
  打开后高亮真实来源框，不自动发起局部分析，也不从线索文本推测坐标。
- `useImmersiveLayout` 通过 `ResizeObserver` 测量实际面板位置；`GeoMap.viewportInsets`
  将这些尺寸用于地图取景及控件避让。手机底部面板仅占用底部安全范围。
  面板变化更新 MapLibre 的持久 padding，不因普通文字或回执更新重新聚焦。

## 网络与隐私

- 默认地图瓦片由 Esri World Imagery 通过浏览器直接请求；显式覆盖时则由所选
  MapLibre 样式提供方或 MapTiler 直接提供瓦片。
- 应用**不会**把上传图片或用户照片发送给底图服务；请求只包含瓦片 URL、样式
  与常规浏览器信息。
- 若瓦片网络被拦截或样式加载失败，地图区域显示 “底图加载失败” 提示，国家/城市
  候选列表仍然可用。

## 交互行为

- `NavigationControl`：缩放/指南针；容器尺寸变化由 MapLibre `trackResize`
  （ResizeObserver）自动处理。
- 国家标记来自 `scores` 的代表点；**双击国家标记**确认国家（与候选列表按钮同一
  行为），红色表示用户已确认的国家。
- 城市标记来自可选的 `city_scores`；`新增按钮/下拉框`按国家请求城市候选，标记为
  城市代表点。
- 所有标记都是目录代表点，**不是照片 GPS，也不是精确定位**；`fitBounds` 使用
  真实 lng/lat，并对跨 180° 经线做 unwrap，避免框选整个地球。
- 坐标缺失、越界、非有限或 (0,0) 的标记会被过滤，不会把国家画到几内亚湾。
- 组件卸载时移除全部 marker 并 `map.remove()`；样式/瓦片错误通过 `error` 事件
  进入可见的降级提示。

## 契约（前端）

- `CountryScore` 代表点在 `lat`/`lng`；可选 `city_scores: CityScore[]`
  （`city_id/iso/name/region/lat/lng/raw_cosine/score/score_type`）。
- 城市 API：`POST /api/geo/{session_id}/cities`，body `{iso}`，返回
  `{city_scores, country_iso, scope, status}`；`status` 为
  `ok | unavailable | no_cities`。
- 后端尚未实现该接口时（404/405），前端清空城市候选并提示“城市候选服务不可用”，
  国家候选与确认流程不受影响。
- 旧城市浏览接口继续只读；新引导流程中的城市确认由对话事件接口完成，城市分数不代表
  照片位置概率。


## 引导流程中的地图

地图直接使用引导会话返回的当前候选与已确认路径，公开流程为国家 → 城市。
选择候选会高亮并移动到其代表点，明确确认后按新层级候选缩小地图范围；
回退、恢复排除和更换图片会清理失效选择，并按最新会话状态重新聚焦。
仅线索说明或回执文字变化不会反复打断用户手动平移和缩放。

无坐标候选仍可在列表中选择，地图仅显示真实已确认父级的代表点作为视野回退，
不给无坐标子级伪造标记或边界。

局部图片分析的范围、原图文字框、来源状态、场景候选和核实建议见
[交互与局部分析协议](interaction-analysis.md)。
