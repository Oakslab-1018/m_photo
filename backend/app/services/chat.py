"""Existing transaction-safe orchestration plus a pure, pre-network conversation preview."""
from __future__ import annotations

from copy import deepcopy
import json
import logging
from types import SimpleNamespace

from services.chat import ChatService as LegacyChatService, format_feedback_receipt
from ..config import Settings
from ..domain.geolocation.confidence import route_confidence
from ..domain.geolocation.model_ranking import ModelAwareEngine, ModelAwareHypotheses
from .deepseek import DeepSeekClient, narration_matches_action
from .model_gateway import StreetClipGateway

logger = logging.getLogger(__name__)


class _PreviewSessions:
    """In-memory planning only: never an ORM or a database write."""
    def __init__(self, snapshot, messages):
        self.row = deepcopy(snapshot)
        self.messages = deepcopy(messages)

    def get(self, session_id):
        return self.row

    def set_clues(self, session_id, clues):
        self.row.clues = deepcopy(clues)

    def add_clue(self, session_id, clue):
        self.row.clues.append(deepcopy(clue))

    def add_exclusion(self, session_id, iso):
        if iso not in self.row.exclusions:
            self.row.exclusions.append(iso)

    def remove_exclusion(self, session_id, iso):
        self.row.exclusions = [item for item in self.row.exclusions if item != iso]

    def remove_user_clues(self, session_id, *, dimension=None, value=None):
        self.row.clues = [
            clue for clue in self.row.clues
            if not (isinstance(clue, dict) and clue.get("source") == "user"
                    and (dimension is None or clue.get("dimension") == dimension)
                    and (value is None or clue.get("value") == value))
        ]

    def set_confirmed(self, session_id, iso):
        self.row.confirmed_iso = iso
        self.row.status = "done"

    def add_message(self, session_id, role, content, meta=None):
        self.messages.append({"id": len(self.messages) + 1, "role": role,
                              "content": content, "meta": meta or {}, "created_at": None})
        return self.messages[-1]

    def list_messages(self, session_id):
        return deepcopy(self.messages)


def _signature(result):
    return json.dumps({"scores": result.get("scores"), "action": result.get("action")},
                      sort_keys=True, ensure_ascii=False, allow_nan=False)


class ChatService(LegacyChatService):
    def __init__(self, session_service, settings: Settings | None = None):
        super().__init__(session_service)
        self.engine = ModelAwareEngine()
        self.hypothesis = ModelAwareHypotheses(self.engine)
        self.provider = DeepSeekClient(settings) if settings is not None else None
        self._prepared = None

    def _settings(self) -> Settings:
        return self.provider.settings if self.provider else Settings.from_env({})

    def _validated_prior(self, evidence):
        if evidence is None:
            return None
        try:
            return StreetClipGateway(self._settings()).validate(evidence)
        except Exception as exc:
            logger.warning("Stored StreetCLIP evidence invalid; using rules (%s)", type(exc).__name__)
            return None

    def _set_prior(self, session_id):
        row = self.session_service.get(session_id)
        self.engine.evidence = self._validated_prior(row.model_evidence) if row else None

    def _finish(self, result, session_id):
        if self._prepared and result.get("action") and _signature(result) == self._prepared[0]:
            message = self._prepared[1]
            action = result["action"]
            payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
            # The deterministic receipt is the acknowledgment of what the
            # user's turn was understood to mean; remote narration may add
            # wording but must never erase it, so it prefixes the final text.
            prefix = format_feedback_receipt(payload.get("feedback_receipt"))
            if prefix and not message.lstrip().startswith(prefix):
                message = f"{prefix}\n{message}"
            result["action"] = {**action, "message": message}
            self.session_service.replace_last_ai_message(session_id, message, result["action"])
        self._prepared = None
        self._free_text_plan = None
        return result

    @staticmethod
    def _apply_confidence_route(result, model_evidence):
        scores = result.get("scores") or []
        route = route_confidence(scores, model_evidence).to_dict()
        action = dict(result.get("action") or {})
        payload = dict(action.get("payload") or {})
        top3 = list(route.get("top3") or [])
        top = top3[0] if top3 else {}
        payload.update({
            "confidence": route,
            "candidate_details": top3,
            "candidates": [str(item.get("iso")) for item in top3 if item.get("iso")],
        })
        if route["tier"] == "high":
            action = {
                "kind": "result",
                "message": (
                    f"高置信判断：{top.get('name_zh') or top.get('iso') or '候选国家'}。"
                    "模型全图排序与融合排序方向一致；这是候选排序，不是概率。"
                ),
                "payload": {
                    **payload,
                    "iso": top.get("iso"),
                    "name_zh": top.get("name_zh"),
                    "score": top.get("score"),
                },
            }
        elif route["tier"] == "medium":
            action = {
                "kind": "confirm",
                "message": (
                    f"候选最可能是「{top.get('name_zh') or top.get('iso') or '未知'}」，"
                    "但也存在接近候选，请确认或从候选中选择。"
                ),
                "payload": {
                    **payload,
                    "iso": top.get("iso"),
                    "name_zh": top.get("name_zh"),
                    "score": top.get("score"),
                },
            }
        else:
            dimensions = payload.get("dimensions") or ["language", "place_name", "driving_side"]
            options = payload.get("options")
            action = {
                "kind": "ask" if action.get("kind") in {"ask", "request_clue"} else "request_clue",
                "message": "当前候选接近，暂不自动确定国家。请补充画面中的语言、招牌文字、道路方向或其他可见线索。",
                "payload": {
                    **payload,
                    "dimensions": dimensions,
                    "options": options,
                    "reason": route["reason"],
                },
            }
        result["action"] = action
        result["confidence"] = route
        return result

    def _route_and_sync(self, result, session_id):
        action = result.get("action") or {}
        if result.get("done") or action.get("kind") == "done" or not self.engine.evidence:
            result["confidence"] = None
            return result
        result = self._apply_confidence_route(result, self.engine.evidence)
        if self._prepared is None:
            action = result.get("action") or {}
            replace = getattr(self.session_service, "replace_last_ai_message", None)
            if callable(replace) and action.get("message"):
                replace(session_id, str(action["message"]), action)
        return result

    def initial_analysis(self, session_id):
        self._set_prior(session_id)
        result = super().initial_analysis(session_id)
        return self._finish(self._route_and_sync(result, session_id), session_id)

    def handle_feedback(self, session_id, feedback):
        self._set_prior(session_id)
        result = super().handle_feedback(session_id, feedback)
        return self._finish(self._route_and_sync(result, session_id), session_id)

    def reason_only(
        self,
        session_id,
        new_clues=None,
        click_pos=None,
        point_analysis=None,
        point_analysis_input=None,
        point_analysis_cacheable=None,
        point_analysis_provider_signature=None,
    ):
        self._set_prior(session_id)
        result = super().reason_only(
                session_id,
                new_clues,
                click_pos,
                point_analysis=point_analysis,
                point_analysis_input=point_analysis_input,
                point_analysis_cacheable=point_analysis_cacheable,
                point_analysis_provider_signature=point_analysis_provider_signature,
            )
        return self._finish(self._route_and_sync(result, session_id), session_id)

    def _prepare(self, snapshot, history, method, *, feedback=None, new_clues=None, click_pos=None):
        self._prepared = None
        if not self.provider or not (self.provider.settings.deepseek_api_key or "").strip():
            return
        memory = _PreviewSessions(snapshot, history)
        preview = LegacyChatService(memory)
        preview.engine = ModelAwareEngine()
        preview.engine.evidence = snapshot.model_evidence
        preview.hypothesis = ModelAwareHypotheses(preview.engine)
        # The deterministic (plus validated remote) parse plan is computed once
        # in prepare_update; the preview and the transactional run must use the
        # exact same plan or their signatures would diverge.
        preview._free_text_plan = self._free_text_plan
        if method == "initial":
            result = preview.initial_analysis(0)
        elif method == "feedback":
            result = preview.handle_feedback(0, feedback)
        else:
            for clue in new_clues or []:
                memory.add_clue(0, clue)
            result = preview.reason_only(0, new_clues=new_clues, click_pos=click_pos)
        if snapshot.model_evidence:
            result = self._apply_confidence_route(result, snapshot.model_evidence)
        else:
            result["confidence"] = None
        if str((result.get("confidence") or {}).get("tier") or "") == "high":
            return
        message = self.provider.narrate({"action": result.get("action"), "scores": result.get("scores", []),
                                       "clues": memory.row.clues, "history": history,
                                       "feedback": (feedback or {}).get("text", "")})
        if message and narration_matches_action(message, result.get("action") or {}):
            self._prepared = (_signature(result), message)

    def prepare_initial(self, clues, evidence):
        snapshot = SimpleNamespace(clues=deepcopy(clues), exclusions=[], model_evidence=self._validated_prior(evidence),
                                   confirmed_iso=None, status="active")
        self._prepare(snapshot, [], "initial")

    def prepare_update(self, session_id, *, feedback=None, new_clues=None, click_pos=None, model_evidence=None):
        row = self.session_service.get(session_id)
        if row is None:
            return
        # ``model_evidence`` lets the point endpoint preview against the exact
        # crop-extended payload it will persist in the same request.
        raw_evidence = row.model_evidence if model_evidence is None else model_evidence
        snapshot = SimpleNamespace(clues=deepcopy(row.clues), exclusions=list(row.exclusions),
                                   model_evidence=self._validated_prior(raw_evidence),
                                   confirmed_iso=row.confirmed_iso, status=row.status)
        history = self.session_service.list_messages(session_id)
        # End the read transaction before the optional external API request.
        self.session_service.rollback()
        self._free_text_plan = None
        if feedback and feedback.get("type") == "free_text":
            text = feedback.get("text") or ""
            pending = self.parser.pending_from_history(history)
            plan = self.parser.parse(text, pending=pending)
            if plan.confirm_iso:
                # The user explicitly answered yes to a confirm question: this
                # is a deterministic confirmation, never an LLM confirmation.
                self._free_text_plan = (text, plan)
                self._prepared = None
                return
            remote = self._extract_remote_plan(text, history, pending)
            if remote is not None:
                plan = plan.merged(remote)
            self._free_text_plan = (text, plan)
        if feedback and feedback.get("type") == "confirm":
            self._prepared = None
            return
        self._prepare(snapshot, history, "feedback" if feedback is not None else "point",
                      feedback=feedback, new_clues=new_clues, click_pos=click_pos)

    def _extract_remote_plan(self, text, history, pending):
        """Optional strict-JSON extraction; invalid payloads degrade to rules.

        Runs before the write transaction, never mutates the session, and the
        validated result is only ever merged as extra clues/rejections (never a
        confirmation) by ``ChineseFreeTextParser.validate_remote``.
        """
        if self.provider is None:
            return None
        if not (self.provider.settings.deepseek_api_key or "").strip():
            return None
        try:
            payload = self.provider.extract({
                "text": text,
                "history": history,
                "pending": pending.to_dict() if pending is not None else None,
                "catalog": self.parser.remote_catalog(),
            })
        except Exception as exc:  # defensive: provider errors never break dialogue
            logger.warning("DeepSeek extraction unavailable; using rules (%s)", type(exc).__name__)
            return None
        if not isinstance(payload, dict):
            return None
        validated = self.parser.validate_remote(payload, text, pending)
        return validated if (validated.clues or validated.reject_isos) else None
