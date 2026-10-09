# 交互式地图（MapLibre GL JS）

## 配置

地图样式通过 Vite 的 public build-time 变量注入，全部会内联进前端产物，
**因此都不是服务端密钥**：

| 变量 | 说明 |
| --- | --- |
| `VITE_MAP_STYLE_URL` | 完整 MapLibre style JSON URL，优先级最高 |
| `VITE_MAPTILER_API_KEY` | MapTiler public key；自动解析为 `https://api.maptiler.com/maps/streets-v4/style.json?key=…` |

- 本地：复制 `frontend/.env.example` 为 `frontend/.env.local` 后填写。
- 容器：`Dockerfile.frontend` 已声明同名 build ARG/ENV，Compose build args 由集成阶段接线。
- 两者都为空时回落到官方 demo tiles
  （`https://demotiles.maplibre.org/style.json`）：**仅开发预览、无生产 SLA**，
  页面左上角会显示 “Demo 底图” 标记；生产构建必须配置上面的变量之一。
- MapTiler key 是公开的客户端 key，必须在 MapTiler 控制台配置允许的来源
  （origins/referrers）限制；不要在 `VITE_*` 中放任何服务端密钥。

## 网络与隐私

- 地图瓦片/样式由第三方底图服务（MapTiler 或 MapLibre demo）通过浏览器直接请求。
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
- 所有标记都只是候选代表点，**不是照片 GPS，也不是精确定位**；`fitBounds` 使用
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
- 后端契约落地前，城市标记仅作可选展示；**不存在城市级确认**，城市分数不代表
  照片位置概率。
