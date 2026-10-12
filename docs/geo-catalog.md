# 地理目录入库与 GeoJSON

国家、省级行政区、城市的唯一分发目录是 `data/geo_catalog.sqlite3`，上传图片保存在运行时上传目录。
后端首次启动在 Alembic 迁移后将 SQLite 快照导入运行数据库；已初始化的目录只核对库内版本和记录数。
正常启动及请求不读取三份原目录 JSON。StreetCLIP 使用同一 SQLite 分发目录和逻辑版本。
显式更新导入在一个事务中完成，失败不提交半份目录，也不影响已有会话和上传图片。

## 存储

- `geo_catalog_source`：源文件 SHA-256、记录数、导入策略版本与时间。
- `geo_place`：层级、稳定 ID、原始来源 ISO、归属国家、父级、名称和别名、坐标及坐标来源。
- 国家 ID 使用 ISO；省级 ID 使用 `source_iso.code`，保留代码前导零；城市 ID 使用十进制字符串。
- 按层级、国家、来源范围和父级建立索引，城市查询先限定范围再分页。
- 中国候选继续按现有策略归并，保留 CN/HK/MO/TW 的来源范围。

所有运行路径均使用数据库，不再提供目录 JSON 回退。
`M_PHOTO_GEO_CATALOG_SEED` 可指定后端初始化快照；`STREETCLIP_CATALOG_DB` 可指定推理目录。
目录 revision 读取库内保存的逻辑版本，保留迁移前的城市版本；国家推理版本仍由候选及模板计算。
SQLite 文件的物理 hash 不作为推理版本。

## 坐标

国家及城市使用已有目录代表坐标。省级源目录不含经纬度，采用
`admin1-largest-city-v1`：在同一来源 ISO 和行政区代码下，取人口最多的目录城市坐标，
人口相同则以城市 ID 排序。保存 `coordinate_city_id`，明确标注为派生代表点。
它不是行政区几何中心、边界或照片 GPS。

无法匹配城市的行政区仍保存在数据库中，坐标保持空值，并标记 `unavailable`。
GeoJSON 只返回具有有效经纬度的 Feature，同时报告 `missing_coordinates`。
不会用 `(0,0)`、国家代表点或虚构坐标补齐。

当前源目录包含 250 个国家/地区、3,865 个一级行政区、34,156 个城市。
2,764 个一级行政区可得到上述代表点；1,101 个尚缺坐标。
如需全部行政区均出图，需要另外提供按行政区 ID 对齐、具有来源和许可的坐标补充数据。

## GeoJSON 接口

- `GET /api/geo/catalog/countries.geojson`
- `GET /api/geo/catalog/admin1.geojson?country=CN`
- `GET /api/geo/catalog/cities.geojson?admin1_id=CN.02&limit=100&offset=0`

查询参数包括 `country`、`source_iso`、`admin1_id`、`q`、`bbox`、`limit` 和 `offset`。
城市必须至少限定国家、行政区或 bbox，避免一次传送全世界城市。
`limit` 默认 500，最大 2,000；分页顺序稳定。
`bbox` 使用 `minLng,minLat,maxLng,maxLat`；第一版不接受跨日期变更线的 bbox。
国家/来源/行政区父子关系不一致时返回 422。

返回 `FeatureCollection`，每个 Feature 的 Point 坐标严格为 `[lng, lat]`，
properties 同时保留数值型 `lat` 和 `lng`，以及层级、父级和坐标来源。

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "id": "JP.40",
    "geometry": {"type": "Point", "coordinates": [139.0, 35.0]},
    "properties": {
      "id": "JP.40", "level": "admin1", "label": "Tokyo",
      "country_iso": "JP", "source_iso": "JP",
      "lat": 35.0, "lng": 139.0,
      "coordinate_source": "admin1-largest-city-v1", "coordinate_quality": "derived"
    }
  }],
  "meta": {
    "total": 47, "returned": 1, "missing_coordinates": 0,
    "limit": 1, "offset": 0, "truncated": true
  }
}
```

上例仅说明字段与坐标顺序，具体坐标以接口返回的目录记录为准。
`meta` 还提供查询范围和各目录逻辑 revision。三份原目录 JSON 已由 SQLite 快照替代。
`geo-catalog-sources.json` 和 `admin1-sources.json` 保留上游来源、许可及溯源信息。

## 维护

更新脚本直接更新 SQLite 分发目录，不再生成三份目录 JSON：

```powershell
python scripts/update-geo-catalog.py --database data/geo_catalog.sqlite3
python scripts/update-admin1-catalog.py --database data/geo_catalog.sqlite3
```

替换快照后，显式将新版本导入已存在的后端运行库：

```powershell
uv run --frozen python -m backend.app.services.geo_catalog_import --force
```

只有需要从外部旧数据迁移时才使用 `scripts/import-legacy-geo-catalog.py`，并显式提供三个 JSON 输入路径；
应用启动不会调用该工具。
