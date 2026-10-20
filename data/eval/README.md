# 离线精度评测

用有标注照片评估国家排序、OCR 融合和城市链路。单元测试只证明代码可运行，不衡量准确率；流程测试通过也不等于定位精度提升。

> 数据：本地 img2gps3k 共 2997 条合法坐标与图片（目录 3000 图，另 3 张未标注），其中 2737 条有派生国家标签、260 条 `country_iso=null`。国家指标只覆盖有标签子集；e2e 城市距离使用全部合法坐标样本。
>
> 状态：confidence/guided 的 backend/frontend 实现进行中，contract/integration tests 待 primary 接受；本文不声明对话准确率、coverage 或问题数收益。

## 已有结果

### Legacy baseline（partial）

- 运行由用户停止在 `1764/2997`；完成的是 manifest/CSV 前缀，不是随机抽样，也不是完整 benchmark。剩余 1233 条未运行，不计作 miss。
- 前缀中有国家标签的 attempted 样本为 1627；country top1 为 `877/1627 = 53.9%`。该数值不能外推为全量结果。
- 报告：[partial-baseline/analysis.md](../../reports/img2gps3k-run-20261020T195456/partial-baseline/analysis.md)。

### 30 图 paired

固定 30 图中有 28 张可评国家标签。三组 `model_only`、`legacy_ocr`、`revised_ocr` 共用同一 full-image prior；不含 crop、reference、城市、对话或 LLM，仅是 country-only 对照。

| 方案 | top1 | top3 | top5 |
| --- | ---: | ---: | ---: |
| 纯 StreetCLIP (`model_only`) | 8/28 (28.6%) | 12/28 (42.9%) | 17/28 (60.7%) |
| 旧 OCR 融合 (`legacy_ocr`) | 8/28 (28.6%) | 13/28 (46.4%) | 17/28 (60.7%) |
| 新 OCR 融合 (`revised_ocr`) | 8/28 (28.6%) | 13/28 (46.4%) | 17/28 (60.7%) |

新 OCR 相比纯模型 top3 多 1 张，top1 无提升；新相比旧也无 top1 变化。该样本量不能证明整体收益，更不能作为对话收益。30 图还与 prospective 1233 holdout 重叠 10 张，因此该 holdout 并非完全 blind。报告：[immediate-smoke30-comparison.md](../../reports/ocr-evidence-ablation-20261020T205803/immediate-smoke30-comparison.md)。

### Full paired（557 条 recovered prefix）

557 条 partial cache 已完成恢复审计和 offline paired replay。`meta.json` 明确标记为 derived；原始 record bytes 未改，并用 manifest、图片、catalog、scorer 和源码 hash 校验。500 条可评国家标签、57 条 unknown-label 排除、0 arm failures；这是 manifest-order prefix，不是随机或 blind holdout。

- top1：三组均为 49.2%。
- top3：model-only 68.2%，legacy OCR 68.6%，revised OCR 68.4%。
- top5：model-only 75.0%，legacy/revised OCR 74.6%。
- revised vs legacy：top1 `3 corrected / 3 harmed`；top3 `1 corrected / 2 harmed`。不宣称准确率提升。
- HIGH `>= 0.02`（revised）：134 eligible，24 wrong，17.9% false-high；只是描述性诊断，阈值 provisional 且未校准。

报告：[full-paired-recovered557-replay/report.md](../../reports/ocr-evidence-ablation-20261020T205803/full-paired-recovered557-replay/report.md)。full 2997 续跑已获授权，剩余 2440 条实现待启动；当前仍未完成，本文不提供恢复/续跑命令。

## 复现

前提：StreetCLIP 已启动且评测配置可用；脚本在宿主机运行，不经过后端 HTTP 服务。

从 img2gps3k CSV 生成审计 manifest：

```powershell
.venv\Scripts\python.exe scripts\import_img2gps3k.py `
    --csv data\eval\images\img2gps3k\im2gps3k.csv `
    --image-root data\eval\images\img2gps3k\images `
    --boundaries .cache\eval-boundaries\naturalearth-5.1.1-admin0.geojson `
    --boundary-sha256 c67e99776f955a9662b53c8419204b7df8ecc52c95d509271435cf3b359aaef5 `
    --boundary-source https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip `
    --output data\eval\manifest.img2gps3k.jsonl `
    --audit-dir reports\img2gps3k-import `
    --catalog-db data\geo_catalog.sqlite3 `
    --retain-unlabeled-country
```

国家/城市评测：

```powershell
.venv\Scripts\python.exe scripts\eval_geolocation.py `
    --manifest data\eval\manifest.img2gps3k.jsonl `
    --allow-unlabeled-country `
    --no-cities `
    --strict `
    --label baseline-img2gps3k
```

`--no-cities` 只关闭 oracle 城市，e2e 城市仍会运行。报告默认写入 `reports/eval-<UTC>/`，也可用 `--output DIR` 指定；`report.md` 供阅读，`report.json` 含完整配置与样本，`samples.jsonl` 增量保存已完成样本。完整参数见脚本 `--help`。

## Manifest 最小格式

JSON Lines，每行一张图；示例见 [`manifest.example.jsonl`](manifest.example.jsonl)。

```json
{"id":"tokyo-001","image":"images/tokyo-001.jpg","country_iso":"JP","lat":35.6895,"lng":139.6917,"city_id":1850147}
```

| 字段 | 必要性 | 说明 |
| --- | --- | --- |
| `id` | 必需 | 唯一标识，匹配 `[A-Za-z0-9._-]+` |
| `image` | 必需 | 相对评测图片根的路径；禁止绝对路径和 `..` |
| `country_iso` | 默认必需 | 两字母来源 ISO；TW/HK/MO 在国家指标中归并为 CN。`--allow-unlabeled-country` 下允许 `null` |
| `lat` / `lng` | 必需 | 标注拍摄点坐标 |
| `city_id` | 建议 | GeoNames 城市 id；提供后才计算精确城市命中 |
| `city_source_iso` | 按需 | 城市检索来源 ISO 与 `country_iso` 不同时填写 |

`city`、`admin1_code`、`tags`、`license`、`source`、`notes` 可选。图片仅支持 jpg/jpeg/png/webp/gif。

## 指标口径

- **国家 `country`**：top-1/3/5、MRR 只以成功产出国家 ranking 的样本为分母。
- **国家 `country.overall`**：以有国家标签的 attempted 样本为分母，失败算 miss；`label_coverage` 单独报告。
- **e2e 城市**：全部合法坐标样本进入距离指标；失败或无候选计入 overall miss，同时报告 available-conditional 和 `available_rate`。
- **oracle 城市**：已知真实国家后只评城市排序，缺国家标签的样本跳过。
- **degraded**：没有 StreetCLIP 证据、只靠规则排序的样本，不能与模型基线混比。
- **margin / accuracy at coverage**：排序分差与覆盖门禁，不是概率。

## 开关与退出

| 参数 | 行为 |
| --- | --- |
| `--no-cities` | 只跳过 oracle 城市；e2e 城市保留 |
| `--no-e2e-cities` | 只跳过 e2e 城市 |
| `--allow-unlabeled-country` | 允许 `country_iso=null`；国家分母为 labelled attempted |
| `--strict` | 样本 error、degraded、启用的城市路径失败等写出报告后退出 1；预检拒绝退出 2 |
| `--min-country-top1 X` | 使用 `country.overall.top1`；低于 `[0,1]` 内的阈值退出 1 |
| `--fail-fast` | 首个样本 error 仍写 partial report，标记 partial，退出 1 |
| `--allow-degraded` | 国家预检失败时整轮禁用模型；仅城市能力失败时保留国家模型 |

## 当前限制

- legacy baseline 只有 1764 条前缀；full paired 的 557 条 recovered prefix 已有离线报告，但 full 2997 正在后台续跑（复用 598 条，计划补采 2399 条后自动回放）。两者都不能替代完整 benchmark。
- 没有真实 reference corpus/index；StreetCLIP 国家排序不是已完成的独立参考检索。国家消融也不含 reference arm。
- LLM 仅做经校验的文字线索抽取与解释，不决定国家/坐标；所有结果仍由同一确定性 scorer 产生，provider 失败回退本地模板。
- 国家消融说明见 [country-ablation.md](../../docs/country-ablation.md)，产品与对话契约见 [guided-dialogue.md](../../docs/guided-dialogue.md)。
