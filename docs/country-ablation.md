# 国家消融：semantic 守卫与 capture/replay

状态：工具已完成第一阶段；尚无真实消融收益结论。全量旧基线仍在运行，城市、对话和 LLM 不在本轮范围内。

## 口径

- 仅评国家；三组 `model_only`、`legacy_ocr`、`revised_ocr` 共用同一 `full_image_only` 完整 prior。replay 评分前会规范化为该视图，丢弃 crop 与 reference sidecar；三组使用同一份 full-image rankings。
- `model_only` 不用线索；`legacy_ocr` 用旧代码 capture 的完整线索；`revised_ocr` 用 capture 的非 semantic 线索加当前 semantic 重新派生。
- ground truth 不写入 cache、不传给模型或打分器。采集只使用 manifest 定位样本；replay 才关联标签计算指标。
- `country_iso=null` 的 unknown country 排除并单列，不进入评分；所有 labelled attempted 样本是同一分母。某组 engine 失败记 miss，同时单列 failure。
- 每个比较输出 top-1/3/5 的 `corrected`、`harmed`、`unchanged` 及 rank transition。只有 paired 结果可判断改动是否有效，不能把单组分数当收益。

## semantic 守卫

旧逻辑会把目录内同名唯一的城市词直接转成国家证据，孤立城市词可抬高某国。新逻辑要求普通城市名在同一 OCR 行有唯一显式国家名，且城市目录 ISO 与之一致；整行恰好是城市名、或目录内同名唯一，都不足以证明现实世界唯一。显式国家名、curated alias 的目录映射保留；通过 canonical country policy 继续兼容 TW/HK/MO 归并到 CN。本次不恢复裸三字母大写词元推断国家。

本轮只改 semantic 判定，不改 scorer/engine 权重或 boost。该守卫可能损失召回，必须用 paired corrected/harmed 检验，不能预设净收益。

## Capture

前置：冻结的旧代码工作区存在，StreetCLIP 与 rapidocr 可用；用原 root 的 env-file、manifest、catalog 和 OCR 模型 cache。`--source-root` 必须是旧代码工作区，不能用当前 worktree 代替。capture 的 `--env-file` 没有默认值；原 root 与 source root 的 `.env` 不同，必须显式传原 root 的 `--env-file`，不能用 source root `.env` 代替。当前约定路径：

```powershell
$PY = 'D:\code\clones\m_photo\.venv\Scripts\python.exe'
$SRC = 'D:\code\clones\m_photo-eval-frozen' # detached commit 7984e34
$ROOT = 'D:\code\clones\m_photo'
$MANIFEST = "$ROOT\data\eval\manifest.img2gps3k.jsonl"
$CATALOG = "$ROOT\data\geo_catalog.sqlite3"
$CACHE = 'D:\code\clones\m_photo-ocr-evidence\reports\country-ablation\img2gps3k-cache'
$env:M_PHOTO_OCR_MODEL_DIR = "$ROOT\backend\storage\ocr-models"

Set-Location 'D:\code\clones\m_photo-ocr-evidence'
& $PY scripts\eval_country_ablation.py capture `
  --source-root $SRC `
  --manifest $MANIFEST `
  --image-root "$ROOT\data\eval" `
  --catalog-db $CATALOG `
  --env-file "$ROOT\.env" `
  --output $CACHE
```

capture 每图各跑一次 OCR 与 full-image prior，写入 `cache.jsonl` 和 `meta.json`。中断时已完成行可留在 `cache.jsonl.partial`；没有最终 `meta.json` 的文件不是完整 cache，不能 replay。需要试跑可在命令加 `--limit 30` 或 `--ids`，但不要复用已占用输出目录。

## Replay

replay 只读 cache、manifest、图片 hash 和当前 worktree 的 scorer/semantic，**不调用 OCR、模型或网络**：

```powershell
$REPLAY = 'D:\code\clones\m_photo-ocr-evidence\reports\country-ablation\img2gps3k-replay'
& $PY scripts\eval_country_ablation.py replay `
  --cache $CACHE `
  --manifest $MANIFEST `
  --image-root "$ROOT\data\eval" `
  --catalog-db $CATALOG `
  --output $REPLAY
```

输出 `report.json`、`report.md`、`meta.json`。replay 会拒绝 model/prompt/catalog/manifest/image 不兼容、cache 校验失败，以及 scorer/engine 字节 hash 与 capture 不一致的 cache；本轮只允许 semantic 变化。输出目录必须新建且与 cache 分离。reference corpus 尚未建立，参考检索、多国分支和最多三问均为后续工作，不应据此文档声称已经实现或已有收益。