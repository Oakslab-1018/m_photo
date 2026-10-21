# 引导对话与置信路由契约

状态：三档路由、线索回流和常驻对话入口已实现并通过相关流程回归。流程通过不代表对话提高了定位准确率、coverage 或问题效率，真实收益仍需离线对照。

## 端到端流程

`上传 → StreetCLIP + OCR → 同一确定性 scorer → canonical margin 路由 → 用户显式确认国家 → city → 结果解释`

- **HIGH**：`action=direct_top1`，显示 predicted top1。它只是模型候选，不是用户确认，也不会推进阶段。
- **MEDIUM**：`action=confirm_candidates`，显示 top3；LLM 可解释，但必须由用户明确选择并确认。
- **LOW**：`action=ask_clue`，最多问 3 个国家线索问题；每次回答重新进入同一 scorer 并重排。预算用完后只能显示显式 uncertain，或让用户从 top3 手动选择；不得自动猜测。
- HIGH/MEDIUM 仅跳过常规逐条观察复核。correction、conflict 或 pinned re-review 仍先阻断 confidence route，必须由用户解决。
- confidence tier 只在 `stage=country` 生效；city 不继承国家 tier，也不能被预测自动确认或推进。

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

- 只有显式 `confirm_location` 能推进：country → city → resolved。排序、HIGH 预测、自由文本和 LLM 都不能确认或推进。
- 确认路径必须有 catalog-valid ancestry；确认被排除地点前要先 retract。
- 排除限定在当前 scope，不能误删其他地区的同名地点。
- `backtrack(country)` 清除确认并回到国家；`backtrack(city)` 清除城市确认并保留国家。旧客户端 `backtrack(admin1)` 按回到国家处理。证据和排除历史保留。
- `retract(target)` 移除排除；`retract(clue_id)` 将观察恢复为 pending。任何使父级失效的操作都清除子级确认。

## 自由文本与 LLM 边界

- 自由文本只生成待校验的 clues、rejections、catalog targets 或 correction proposal，不能直接确认地点。
- LLM 只做 text-only、经 schema/quote/catalog 校验的抽取与 narration；不接收图片、GPS、坐标，不独立决定国家，也不改写 scorer 结果。
- 每条新增/复核线索仍进入同一确定性 scorer；LLM 不能新增排除条件、改变动作或越过问题预算。
- LLM provider 失败、无 key、超时或输出不合规模板时，保留本地模板。只有通过动作校验并实际采用的 DeepSeek 措辞才记录 `conversation_provider=deepseek`；模板同步会清除该标记并保留其他元数据。
- final resolution 的解释同样只能措辞，不能改变已确认路径。

## 候选与前端

- 国家候选来自同一 scorer 的 canonical top3；HIGH 只显示 top1，MEDIUM/LOW 可显示 top3。不得把 catalog-only 候选伪装成有模型分数。
- 确认国家后直接检索该国城市；CN 按现有国家策略覆盖 CN/HK/MO/TW 来源。省份只保留为城市目录属性，不参与新流程确认，也不限制旧会话的城市范围。仅有实际模型结果时才显示评分。
- “补充线索，与定位助手对话”输入框固定在引导面板底部，与候选列表分开滚动；国家、城市及完成阶段均可提交。使用“发送线索”提交文字，成功后显示回执，失败或冲突保留草稿。
- 前端展示确认路径、当前候选、观察复核、单个问题、排除/恢复、回退和 resolved 结果。新增待复核观察会立即使确认按钮失效，直到 authoritative snapshot 重新加载。
- 409 时重载并提示手动重试；问题 ID/kind/target 改变后不得按旧问题提交。新上传清空引导状态；网络重试复用同一 `event_id`。
- 地图/目录坐标是代表点，不是照片 GPS。

参考索引当前没有真实 corpus/index；StreetCLIP 国家排序不等于已完成的独立参考检索。


## 两步流程与文字地名线索

- 新响应只返回 `country|city|resolved` 阶段；国家确认直接到城市，城市确认到完成。`admin1` target/confirmed 字段仅为旧状态兼容与目录归属保留，不出现省份确认步骤。旧会话按已确认国家/城市恢复，无需改写数据库或重新选省。
- 城市检索覆盖已确认国家的完整目录；用户明确提供的城市、目录检索命中以 `selection_basis=user_text|catalog_search` 标注，优先展示。没有模型评分的匹配仍是目录候选，不伪造余弦。
- 独立或带引号的路名、街区名作为 `signage` 用户文字观察保存，并标记只作展示；不因名字像某个城市就自动提高国家/城市分数。问句、否定、不确定表达和执行指令不变成肯定观察。
- 版本化街区关联表可把已核验字面线索关联到目录城市候选；“玉林路/玉林西路”关联成都只作为待确认建议，不声明同名道路全球唯一。记录来源与目录 ID 校验，缺目录或未知路名仍保存原文字线索，并提示继续补充附近地名。
- 重复提供同一文字不新增重复卡片、不叠加评分；后续拒绝该线索会撤销其候选建议。LLM 只提取有原文引述的文字，不把路名改写成城市或坐标。
