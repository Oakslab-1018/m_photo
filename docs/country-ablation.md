# 国家消融：semantic 守卫与 capture/replay

状态：国家消融工具与已保存结果的文档可复核；关联的 confidence/guided backend/frontend 仍在实现，测试待 primary 接受。本文不作对话准确率、coverage 或问题数收益声明。

## 口径

- 仅评国家。三组 `model_only`、`legacy_ocr`、`revised_ocr` 共用同一 `full_image_only` prior；不含 crop、reference、城市、对话或 LLM。
- `model_only` 不用线索；`legacy_ocr` 使用旧代码 capture 的完整线索；`revised_ocr` 使用旧 capture 的非 semantic 线索加当前 semantic 重新派生。
- ground truth 不写入 cache，也不传给模型或 scorer；只用 manifest 定位样本采集，replay 时才关联标签。
- `country_iso=null` 单列为 unknown 并排除；同一 labelled attempted 分母下，某 arm 失败记 miss，同时单列 failure。
- 每个比较输出 top-1/3/5 的 corrected/harmed/unchanged 与 rank transition；只有 paired 结果可判断改动方向。
- 参考索引代码/sidecar 契约存在，但当前没有真实 reference corpus/index；StreetCLIP 国家排序不是已完成的独立参考索引检索。

## 结果

### Legacy baseline（partial）

用户停止在 `1764/2997`。这是 CSV/manifest 前缀，不是随机抽样或完整 benchmark；剩余的 1233 条未运行，也不计作 miss。前缀中 1627 条有国家标签，country top1 为 `877/1627 = 53.9%`。报告：[partial-baseline/analysis.md](../reports/img2gps3k-run-20261020T195456/partial-baseline/analysis.md)。

### 30 图 paired

28 条可评标签，0 arm failure：

| arm | top1 | top3 | top5 |
| --- | ---: | ---: | ---: |
| model_only | 8/28 | 12/28 | 17/28 |
| legacy_ocr | 8/28 | 13/28 | 17/28 |
| revised_ocr | 8/28 | 13/28 | 17/28 |

revised 相比 legacy 的 top1/3/5 均无净变化，不能声称 OCR 规则带来 top1 提升。报告：[immediate-smoke30-comparison.md](../reports/ocr-evidence-ablation-20261020T205803/immediate-smoke30-comparison.md)。

### 557 条 recovered prefix

原 557 条 partial cache 已完成恢复审计和 offline replay。`meta.json` 明确标记为 derived；原始 record bytes 未改，并通过 manifest、图片、catalog、scorer 与源码 hash 校验。500 条有国家标签、57 条 unknown-label 排除、0 arm failures；仍是 manifest-order prefix，不是随机或 blind holdout。

| arm | top1 | top3 | top5 |
| --- | ---: | ---: | ---: |
| model_only | 49.2% | 68.2% | 75.0% |
| legacy_ocr | 49.2% | 68.6% | 74.6% |
| revised_ocr | 49.2% | 68.4% | 74.6% |

revised vs legacy：top1 `3 corrected / 3 harmed`，top3 `1 corrected / 2 harmed`，top5 `1 corrected / 1 harmed`。没有准确率净增益，不能选为新主策略的依据。

HIGH `>= 0.02` 的描述性诊断（revised）：134 eligible、24 wrong、17.9% false-high。阈值来自 provisional policy，未在此子集调参，也未经完整 benchmark 校准。报告：[full-paired-recovered557-replay/report.md](../reports/ocr-evidence-ablation-20261020T205803/full-paired-recovered557-replay/report.md)。

full 2997 已恢复后台采集，复用 598 条记录、补采 2399 条后自动回放；状态见根目录 reports/ocr-evidence-ablation-20261020T205803/ACTIVE_RUN.json，目前没有完整 benchmark 结论。30 图还与 prospective 1233 holdout 重叠 10 张，因此该 holdout 并非完全 blind。

## Semantic 守卫

旧逻辑会把目录内同名唯一的城市词直接转成国家证据，孤立城市词可抬高某国。新逻辑要求普通城市名在同一 OCR 行有唯一显式国家名，且目录 ISO 与之一致；整行恰好是城市名、或目录内同名唯一，都不足以证明现实世界唯一。显式国家名和 curated alias 的目录映射保留；TW/HK/MO 继续通过 canonical country policy 归并为 CN。本轮不恢复裸三字母大写词元推断国家。

本轮只改 semantic 判定，不改 scorer/engine 权重或 boost。守卫可能损失召回，必须继续看 paired corrected/harmed，不能预设净收益。

## Capture / replay

capture 需要冻结的旧代码 worktree、原 root 的 env-file、manifest、image-root、catalog 和 OCR model cache；`--source-root` 必须指向旧代码，不能用当前 worktree 代替。replay 只读完整 cache、manifest、图片 hash 与当前 scorer/semantic，不调用 OCR、模型或网络。

```powershell
& <python> scripts\eval_country_ablation.py capture `
  --source-root <frozen-worktree> --manifest <manifest.jsonl> `
  --image-root <eval-root> --catalog-db <catalog.sqlite3> `
  --env-file <original-root-.env> --output <new-cache-dir>

& <python> scripts\eval_country_ablation.py replay `
  --cache <complete-cache-dir> --manifest <manifest.jsonl> `
  --image-root <eval-root> --catalog-db <catalog.sqlite3> `
  --output <new-replay-dir>
```

capture 完成前可留下 `cache.jsonl.partial`；只有最终 `meta.json` 才表示正常完成。replay 会拒绝 model/prompt/catalog/manifest/image 不兼容、cache 校验失败，以及 scorer/engine 字节 hash 不一致的 cache；本轮只允许 semantic 变化。输出目录必须新建且与 cache 分离。本文不提供续跑/恢复命令；完整 2997 的状态以 primary 后续报告为准。
