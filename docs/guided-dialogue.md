# 引导对话与置信路由契约

状态：backend/frontend 实现与 contract/integration tests 均待 primary 接受。本文描述目标契约，不声明测试已通过，也不声明对话提升准确率、coverage 或平均问题数。

## 端到端流程

`上传 → StreetCLIP + OCR → 同一确定性 scorer → canonical margin 路由 → 用户显式确认国家 → admin1 → city → 结果解释`

- **HIGH**：`action=direct_top1`，显示 predicted top1。它只是模型候选，不是用户确认，也不会推进阶段。
- **MEDIUM**：`action=confirm_candidates`，显示 top3；LLM 可解释，但必须由用户明确选择并确认。
- **LOW**：`action=ask_clue`，最多问 3 个国家线索问题；每次回答重新进入同一 scorer 并重排。预算用完后只能显示显式 uncertain，或让用户从 top3 手动选择；不得自动猜测。
- HIGH/MEDIUM 仅跳过常规逐条观察复核。correction、conflict 或 pinned re-review 仍先阻断 confidence route，必须由用户解决。
- confidence tier 只在 `stage=country` 生效；admin1/city 不继承国家 tier，也不能被预测自动确认或推进。

## 置信路由

- 先在 canonical country group 上归并 TW/HK/MO 到 CN，再做 margin、leader 和 top3 比较。
- `model_only_margin`：经校验的 StreetCLIP full-image 国家排序 top1−top2。
- `fused_margin`：确定性 scorer 融合 OCR 线索后的国家排序 top1−top2。
- `effective_margin = min(model_only_margin, fused_margin)`；任一 margin 缺失、非有限或排序/校验不一致时，路由为 LOW。leader changed 本身是有效的 MEDIUM 条件。
- HIGH：`effective_margin >= 0.02` 且 model-only 与 fused leader 相同。
- MEDIUM：未达 HIGH，且 `effective_margin >= 0.005` 或 leader changed；leader change 最高只能到 MEDIUM。
- LOW：其余情况。

`0.02/0.005` 是 provisional policy thresholds，不是概率，也尚未校准。557 条 recovered prefix 的描述性诊断见 [recovered557 report](../reports/ocr-evidence-ablation-20261020T205803/full-paired-recovered557-replay/report.md)；该 prefix 不能替代 full benchmark。

## HTTP 与响应

- `POST /api/geo/{session_id}/dialogue/start`：初始化一次；重复调用返回现有状态，不重置确认或重复消息。
- `GET /api/geo/{session_id}/dialogue`：读取当前快照；未初始化返回 404。
- `POST /api/geo/{session_id}/dialogue/events`：原子校验并应用一个事件。

响应：

```json
{
  "dialogue": {},
  "candidates": [],
  "action": {"kind": "result", "message": "...", "payload": {}},
  "confidence": {"tier": "high", "action": "direct_top1", "top3": []},
  "messages": [],
  "scores": [],
  "clues": [],
  "exclusions": [],
  "receipt": null,
  "inference_notice": null
}
```

`confidence` 使用 `ConfidenceRouteModel`：`tier`、`action`、两个 margin、`effective_margin`、`prior_top_iso`、`top1_iso`、`top3`、`leader_changed`、`reason`、`policy_version`。`top3` 是稀疏排序摘要，不是完整国家行。

## 状态与类型

`LocationTarget`：`{level: country|admin1|city, id, label, country_iso, source_iso?, admin1_id?, city_id?, lat?, lng?, coordinate_source?, coordinate_quality?, coordinate_city_id?}`。country id 为 canonical ISO；admin1 id 为 `source_iso.code`；city id 为十进制字符串。ID、父级和目录归属必须由本地 catalog 验证。

`Dialogue`：

```json
{
  "version": 1,
  "revision": 3,
  "stage": "country",
  "confirmed": {"country": null, "admin1": null, "city": null},
  "exclusions": [],
  "clues": [],
  "pending_question": null
}
```

`ReviewedClue`：`{id, dimension, value, description, source, confidence?, status: pending|confirmed|rejected|uncertain}`。原始机器证据不可改写；用户复核状态单独保存。`unknown` 表示无法判断，不表示排除。

`GuidedQuestion`：`kind: review_clue|ask|confirm_location|request_clue`，带稳定 `id`、`message`，以及可选的 `dimension`、`clue_id`、`target`、`options`。每轮只允许一个 principal question。

## 事件与并发

`GuidedEventRequest` 允许：

- `type: confirm_clue|reject_clue|confirm_location|reject_location|answer|unknown|retract|backtrack|free_text`
- 必填 `event_id`、`expected_revision`
- 按事件填写 `question_id`、`target`、`clue_id`、`value`、`text`、`level`

有效事件递增 revision。重复 `event_id` 且语义相同返回原 receipt；同 ID 不同 payload 返回 409。stale revision、已变化的问题或冲突写入返回 409 且不改状态；无效目录 ID、祖级或事件返回 422。客户端遇到 409 必须重载 authoritative snapshot，不能自动重放旧确认。

## 确认、排除与回退

- 只有显式 `confirm_location` 能推进：country → admin1 → city → resolved。排序、HIGH 预测、自由文本和 LLM 都不能确认或推进。
- 确认路径必须有 catalog-valid ancestry；确认被排除地点前要先 retract。
- 排除限定在当前 scope，不能误删其他地区的同名地点。
- `backtrack(country)` 清除全部确认；`backtrack(admin1)` 清除 admin1/city；`backtrack(city)` 清除 city。证据和排除历史保留。
- `retract(target)` 移除排除；`retract(clue_id)` 将观察恢复为 pending。任何使父级失效的操作都清除子级确认。

## 自由文本与 LLM 边界

- 自由文本只生成待校验的 clues、rejections、catalog targets 或 correction proposal，不能直接确认地点。
- LLM 只做 text-only、经 schema/quote/catalog 校验的抽取与 narration；不接收图片、GPS、坐标，不独立决定国家，也不改写 scorer 结果。
- 每条新增/复核线索仍进入同一确定性 scorer；LLM 不能新增排除条件、改变动作或越过问题预算。
- LLM provider 失败、无 key、超时或输出不合规模板时，保留确定性 action/message，即本地模板回退。
- final resolution 的解释同样只能措辞，不能改变已确认路径。

## 候选与前端

- 国家候选来自同一 scorer 的 canonical top3；HIGH 只显示 top1，MEDIUM/LOW 可显示 top3。不得把 catalog-only 候选伪装成有模型分数。
- admin1 来自权威目录，不依赖国家 top-10 城市；city 搜索必须先限定在已确认 admin1 内，再应用 limit。
- 前端展示确认路径、当前候选、观察复核、单个问题、排除/恢复、回退和 resolved 结果。新增待复核观察会立即使确认按钮失效，直到 authoritative snapshot 重新加载。
- 409 时重载并提示手动重试；问题 ID/kind/target 改变后不得按旧问题提交。新上传清空引导状态；网络重试复用同一 `event_id`。
- 地图/目录坐标是代表点，不是照片 GPS。

参考索引当前没有真实 corpus/index；StreetCLIP 国家排序不等于已完成的独立参考检索。