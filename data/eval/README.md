# 离线精度评测

用标注照片评测当前地理定位链路，回答“有多准、何时降级、城市链路损失在哪里”。
单元测试只证明代码可用，不衡量准确率。

> 仓库当前只有 [`manifest.example.jsonl`](manifest.example.jsonl)，没有真实图片或已标注
> manifest；尚无真实 baseline 结果。

## 快速开始

前提：StreetCLIP 已启动且评测配置可用；脚本在宿主机运行，不经过后端 HTTP 服务。

```powershell
# 1. 生成骨架，再逐条填写 country_iso / lat / lng
.venv\Scripts\python.exe scripts\make_eval_manifest.py `
    --image-root data\eval\images --output data\eval\manifest.jsonl

# 2. 运行国家 + oracle 城市 + e2e 城市评测
.venv\Scripts\python.exe scripts\eval_geolocation.py `
    --manifest data\eval\manifest.jsonl --label baseline

# 3. 只评国家：两个开关都要加
.venv\Scripts\python.exe scripts\eval_geolocation.py `
    --manifest data\eval\manifest.jsonl --label country-only `
    --no-cities --no-e2e-cities
```

`--allow-degraded` 允许在模型不可用时继续，结果会标为 `degraded=true`。
完整参数见 `.venv\Scripts\python.exe scripts\eval_geolocation.py --help`。

报告默认写到 `reports/eval-<UTC>/`，也可用 `--output DIR` 指定：

| 文件 | 用途 |
| --- | --- |
| `report.md` | 人读汇总 |
| `report.json` | 配置、指标和完整 `samples` 数组（兼容旧读取） |
| `samples.jsonl` | 逐样本流式写入；partial 时只含已完成样本 |

## Manifest 最小格式

JSON Lines，每行一张图；完整示例见 [`manifest.example.jsonl`](manifest.example.jsonl)。

```json
{"id":"tokyo-001","image":"images/tokyo-001.jpg","country_iso":"JP","lat":35.6895,"lng":139.6917,"city_id":1850147}
```

| 字段 | 必要性 | 说明 |
| --- | --- | --- |
| `id` | 必需 | 唯一标识，匹配 `[A-Za-z0-9._-]+` |
| `image` | 必需 | 相对评测图片根（默认 manifest 父目录）的路径，禁止绝对路径和 `..` |
| `country_iso` | 必需 | 目录中的来源 ISO；TW/HK/MO 在国家指标中归并为 CN |
| `lat` / `lng` | 必需 | 标注拍摄点坐标 |
| `city_id` | 建议 | GeoNames 城市 id；只有提供后才计算精确城市命中 |
| `city_source_iso` | 按需 | 城市检索来源 ISO 与 `country_iso` 不同时填写，TW/HK/MO 常见 |

`city`、`admin1_code`、`tags`、`license`、`source`、`notes` 可选，不参与最小评测闭环。
`image` 仅支持 jpg/jpeg/png/webp/gif。生成器优先让 `image` 相对 manifest 父目录；若外部图片根无法安全相对，会打印必须追加 `--image-root` 的评测命令。

## 指标语义

- **国家 `country`**：top-1/top-3/top-5、MRR 等沿用旧口径，只以成功产出国家 ranking 的样本为分母。
- **国家 `country.overall`**：同字段以全部 attempted 样本为分母，失败算 miss；`coverage` 分母也是
  全部 attempted，`accuracy` 仍只统计 margin 达标、实际回答的样本。
- **accuracy at coverage**：按 top-1 与 top-2 分数差决定是否回答；它不是余弦概率。
- **oracle 城市**：已知真实国家，只评城市排序，衡量城市检索上限。
- **e2e 城市**：用融合后的 top-1 国家再检索城市，衡量完整链路；另看国家答对时的条件表现。
- **城市可用性与 overall**：原 distance/exact/均值仍为 available-conditional，并新增
  `available_rate`（attempted 分母）；`city.true_country.overall` 与 `city.e2e.overall` 均以全部 attempted 为分母，失败算 miss，报
  `top1_at_25/50/100km` 和 `top3_at_50km`。`top1_exact` 仍只在有 `city_id` 的 available
  样本上计算，不另报 overall。
- **城市距离**：预测城市代表点到标注拍摄点的球面距离；没有 `city_id` 仍可评距离档位，
  距离天然含目录离散化误差。
- **degraded**：样本没有 StreetCLIP 证据、仅靠规则排序；不能与模型基线混比。

## 开关与退出行为

| 参数 | 行为 |
| --- | --- |
| `--no-cities` | 只跳过 oracle 城市；e2e 城市仍会运行 |
| `--no-e2e-cities` | 只跳过 e2e 城市 |
| `--strict` | 写出报告后，样本 error、degraded、任一启用的 oracle/e2e 路径或城市目录失败均退出 1；预检直接拒绝则退出 2 且不写报告 |
| `--min-country-top1 X` | 使用 `country.overall.top1`；`X` 必须是 `[0,1]` 内有限数，低于 `X` 退出 1 |
| `--fail-fast` | 首个样本 error 仍写 partial report，标记 `run.partial`、`selected_samples`、`completed_samples`，退出 1 |
| `--allow-degraded` | 国家预检失败时整轮禁用模型、只跑规则，避免混跑；仅 city capability 失败时保留国家模型，城市路径标不可用 |

`--min-country-top1` 不使用 available-conditional 的 `country.top1`。配置、manifest 或预检错误
退出 2；正常完成且未触发门禁退出 0。

## 当前与计划

**当前**

- 评测脚本可跑现有链路：OCR → StreetCLIP → 线索融合 → 国家排序，以及可选 oracle/e2e 城市排序。
- 参考索引的国家评分目前仅作诊断，不参与国家排序；将其纳入统一 scorer 仍属计划，尚无真实参考语料或指标结果。
- 引导对话已能收集和确认线索；`ModelAwareHypotheses` 会找 top-2 的不同特征维度并避免重复，但仍按固定维度优先级，未用 margin/问题预算决定何时停止。guided 主流程仍优先逐条核对线索，再确认地点。
- LLM 当前仅有可选文字抽取/措辞，尚未接入 guided 建议问题；不看图、不独立决定地点。

**计划顺序**

1. **检索基线**：不做对话，先用检索证据和固定 scorer 建立可复现基线。
2. **参考索引进 scorer**：把参考检索作为同一 scorer 的证据参与决策；检索明确时直接回答。
3. **候选决策策略**：在现有 top-2 差异维度基础上，加入有把握时直接回答、按问题价值/预算选择提问；最多 3 问，仍不明确则弃权。
4. **同集 A/B**：在同一 manifest 上对比有/无对话的准确率、coverage、弃权率和平均问题数。
5. **最后接入 LLM**：将建议问题和解释接入 guided；LLM 仍只处理结构化文字，不新增独立的地理决策权。

证据入口：[参考索引实现](../../docs/reference-index.md) · [引导对话契约](../../docs/guided-dialogue.md) · [评测脚本](../../scripts/eval_geolocation.py)
