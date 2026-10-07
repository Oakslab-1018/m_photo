"""人机协同多轮对话服务。

实现 "AI 线索输出 -> 用户反馈 -> AI 迭代分析" 的假设-验证循环。
支持的用户反馈类型：
- confirm  : 确认某国家（设置 confirmed_iso，结束会话）
- reject   : 排除某国家（加入 exclusions，重新推理）
- answer   : 回答某维度特征（追加线索，重新推理）
- free_text: 自由文本，尝试解析为线索或反馈
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from reasoning.clue import Clue, DIM_LABELS, VALUE_LABELS, dicts_to_clues
from reasoning.engine import ReasoningEngine
from reasoning.hypotheses import HypothesisManager
from reasoning import knowledge_graph as kg


class ChatService:
    def __init__(self, session_service):
        self.session_service = session_service
        self.engine = ReasoningEngine()
        self.hypothesis = HypothesisManager(self.engine)

    def _clues_of(self, session) -> List[Clue]:
        return dicts_to_clues(session.clues)

    @staticmethod
    def _country_name(iso: str) -> str:
        country = kg.get_by_iso(iso)
        return country["name_zh"] if country else iso

    def initial_analysis(self, session_id: int) -> Dict[str, Any]:
        """AI 首次分析后生成初始假设动作。"""
        session = self.session_service.get(session_id)
        if not session:
            return {"error": "session not found"}
        clues = self._clues_of(session)
        scores = self.engine.infer(clues, exclusions=session.exclusions)
        action = self.hypothesis.next_action(scores, clues)
        self.session_service.add_message(
            session_id, "ai", action.message,
            meta={"action": action.to_dict(), "scores": [s.to_dict() for s in scores]},
        )
        return {
            "scores": [s.to_dict() for s in scores],
            "action": action.to_dict(),
        }

    def handle_feedback(self, session_id: int, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """处理用户反馈并迭代推理。

        feedback 结构：
        {"type": "confirm"|"reject"|"answer"|"free_text", "iso"?, "dimension"?, "value"?, "text"?}
        """
        session = self.session_service.get(session_id)
        if not session:
            return {"error": "session not found"}

        ftype = feedback.get("type")
        user_msg = feedback.get("text") or ""

        if ftype == "confirm":
            iso = feedback.get("iso")
            if iso:
                self.session_service.set_confirmed(session_id, iso)
                cname = self._country_name(iso)
                self.session_service.add_message(session_id, "user", f"确认：{cname}")
                self.session_service.add_message(
                    session_id, "ai",
                    f"已锁定拍摄地为「{cname}」。推理完成！",
                    meta={"action": {"kind": "done", "iso": iso}},
                )
                return {"done": True, "iso": iso,
                        "messages": self.session_service.list_messages(session_id)}

        elif ftype == "reject":
            iso = feedback.get("iso")
            if iso:
                self.session_service.add_exclusion(session_id, iso)
                cname = self._country_name(iso)
                self.session_service.add_message(session_id, "user", f"排除：{cname}")

        elif ftype == "answer":
            dim = feedback.get("dimension")
            value = feedback.get("value")
            if dim and value:
                self.session_service.add_clue(session_id, {
                    "dimension": dim,
                    "value": value,
                    "confidence": 0.9,
                    "evidence": "用户确认",
                    "source": "user",
                })
                dim_zh = DIM_LABELS.get(dim, dim)
                val_zh = VALUE_LABELS.get(value, value)
                self.session_service.add_message(
                    session_id, "user", f"补充线索：{dim_zh} = {val_zh}")

        elif ftype == "free_text":
            # 简单关键词解析：识别常见国家名、语言、植被等
            self.session_service.add_message(session_id, "user", user_msg)
            parsed = self._parse_free_text(user_msg)
            for clue in parsed:
                self.session_service.add_clue(session_id, clue)

        # 重新推理
        session = self.session_service.get(session_id)  # 刷新
        clues = self._clues_of(session)
        scores = self.engine.infer(clues, exclusions=session.exclusions)
        action = self.hypothesis.next_action(scores, clues)

        self.session_service.add_message(
            session_id, "ai", action.message,
            meta={"action": action.to_dict(), "scores": [s.to_dict() for s in scores]},
        )
        return {
            "scores": [s.to_dict() for s in scores],
            "action": action.to_dict(),
        }

    def reason_only(self, session_id: int,
                    new_clues: Optional[List[Dict[str, Any]]] = None,
                    click_pos: Optional[tuple] = None) -> Dict[str, Any]:
        """仅根据当前线索重新推理（用于局部点击分析后更新评分）。

        new_clues: 本次局部分析新发现的线索，用于在对话中给出分析依据。
        click_pos: (x_ratio, y_ratio) 点击位置，用于消息中说明。
        """
        session = self.session_service.get(session_id)
        if not session:
            return {"error": "session not found"}
        clues = self._clues_of(session)
        scores = self.engine.infer(clues, exclusions=session.exclusions)
        action = self.hypothesis.next_action(scores, clues)

        # 构造分析依据消息
        pos_text = ""
        if click_pos:
            pos_text = f"你点击了图片 {click_pos[0]*100:.0f}%, {click_pos[1]*100:.0f}% 的位置，"

        basis_lines = []
        if new_clues:
            basis_lines.append("该区域分析出以下线索：")
            for c in new_clues:
                dim_zh = DIM_LABELS.get(c.get("dimension"), c.get("dimension"))
                val_zh = VALUE_LABELS.get(c.get("value"), c.get("value"))
                conf = c.get("confidence", 0)
                basis_lines.append(f"• {dim_zh}：{val_zh}（置信度 {conf*100:.0f}%）")
        if scores:
            top = scores[0]
            basis_lines.append(f"\n基于以上线索，当前最可能的国家是「{top.name_zh}」。")

        msg = f"{pos_text}已完成局部区域分析。" + "\n".join(basis_lines) + f"\n{action.message}"

        self.session_service.add_message(
            session_id, "ai", msg,
            meta={"action": action.to_dict(), "scores": [s.to_dict() for s in scores]},
        )
        return {
            "scores": [s.to_dict() for s in scores],
            "action": action.to_dict(),
        }

    # ---- 自由文本解析（启发式） ----
    _LANG_MAP = {
        "日语": ("language", "ja"), "日文": ("language", "ja"),
        "中文": ("language", "zh"), "汉语": ("language", "zh"),
        "韩语": ("language", "ko"), "韩文": ("language", "ko"),
        "俄语": ("language", "cyrillic"), "俄文": ("language", "cyrillic"),
        "阿拉伯语": ("language", "arabic"), "泰语": ("language", "thai"),
        "英文": ("language", "latin"), "法语": ("language", "latin"),
        "德语": ("language", "latin"), "西班牙语": ("language", "latin"),
    }
    _VEG_MAP = {
        "森林": ("vegetation", "temperate_forest"),
        "草原": ("vegetation", "grassland"),
        "沙漠": ("vegetation", "desert"),
        "雨林": ("vegetation", "tropical_rainforest"),
        "稀树草原": ("vegetation", "savanna"),
    }
    _CLIMATE_MAP = {
        "地中海": ("climate", "mediterranean_sunny"),
        "温带海洋": ("climate", "temperate_maritime"),
        "寒冷": ("climate", "cold_polar"),
        "干旱": ("climate", "arid"),
        "温带": ("climate", "temperate"),
    }

    def _parse_free_text(self, text: str) -> List[Dict[str, Any]]:
        clues = []
        for kw, (dim, val) in self._LANG_MAP.items():
            if kw in text:
                clues.append({"dimension": dim, "value": val, "confidence": 0.8,
                              "evidence": "用户文本", "source": "user"})
        for kw, (dim, val) in self._VEG_MAP.items():
            if kw in text:
                clues.append({"dimension": dim, "value": val, "confidence": 0.7,
                              "evidence": "用户文本", "source": "user"})
        for kw, (dim, val) in self._CLIMATE_MAP.items():
            if kw in text:
                clues.append({"dimension": dim, "value": val, "confidence": 0.7,
                              "evidence": "用户文本", "source": "user"})
        return clues
