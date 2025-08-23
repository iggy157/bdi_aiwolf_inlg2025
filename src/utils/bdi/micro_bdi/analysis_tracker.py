"""analysis.ymlを生成するためのモジュール（LLM主体+役職語フィルタ版）."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
import hashlib
import time
import random
import re
import os

import yaml
import ulid
from dotenv import load_dotenv
from pydantic import SecretStr
from tempfile import NamedTemporaryFile

# LLM backends (fallback 用)
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Talk


# ====== 定数（プロンプトは config 側で一括管理：本ファイルに埋め込まない） ======
DEFAULT_FIXTURE_UTTERANCES = [
    "Are there any seer claims?",
    "Please give a tentative vote with a reason.",
    "Share one town and one wolf read.",
    "Who do you think is suspicious and why?",
    "What is your current analysis of the situation?",
]

# 役職ラベル（英/日）。co判定はLLMに任せるが、役職語が本文に存在しない場合はcoを不許可にするフィルタを行う
ROLE_WORDS_EN = ["villager", "seer", "werewolf", "possessed", "bodyguard", "medium"]
ROLE_WORDS_JA = ["村人", "占い師", "人狼", "狂人", "騎士", "霊媒師"]
ROLE_WORDS_ALL = ROLE_WORDS_EN + ROLE_WORDS_JA

# 役職語フィルタ用の正規表現（英は単語境界、日本語はそのまま）
_RE_ROLE_WORD = re.compile(
    r"\b(?:villager|seer|werewolf|possessed|bodyguard|medium)\b|(?:村人|占い師|人狼|狂人|騎士|霊媒師)",
    re.IGNORECASE,
)

# 疑い/投票（negative）や擁護/好意（positive）に出やすい語（ヒューリスティック補助）
NEGATIVE_HINTS = [
    "suspicious", "suspect", "vote", "lynch", "eliminate", "execute",
    "怪しい", "疑う", "黒", "人狼", "吊りたい", "投票", "入れる", "処刑"
]
# ※ CO近傍語の誤爆を避けるため、役職語は含めない
POSITIVE_HINTS = [
    "trust", "agree", "clear", "innocent", "support", "defend",
    "信用", "賛成", "白", "人間", "好感", "心強い", "擁護", "かばう"
]


def _resolve_project_root(start: Path) -> Path:
    """プロジェクトルートを解決（config/.env を探索）。"""
    for p in [start] + list(start.parents):
        if (p / "config" / ".env").exists():
            return p
    parents = list(start.parents)
    return parents[3] if len(parents) >= 4 else (parents[-1] if parents else start)


def _safe_game_timestamp_from_ulid(game_id: str) -> str:
    """ULID→時刻の安全変換（失敗時は現在時刻）。"""
    try:
        u: ulid.ULID = ulid.parse(game_id)
        tz = datetime.now(UTC).astimezone().tzinfo
        return datetime.fromtimestamp(u.timestamp().int / 1000, tz=tz).strftime("%Y%m%d%H%M%S%f")[:-3]
    except Exception:
        return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]


def _strip_code_fences(s: str) -> str:
    """ ```yaml ...``` / ```json ...``` / ``` ...``` を剥がす。"""
    if not s:
        return s
    m = re.search(r"```(?:yaml|yml|json)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
    return m.group(1).strip() if m else s


def _parse_credibility_scores(resp: str) -> dict[str, float]:
    """信頼度応答の頑健パーサ（YAML/JSON/素の key: value 行/ラベル揺れ対応）。"""
    if not resp:
        return {}
    s = _strip_code_fences(resp).strip()

    # 1) YAML/JSONとして試行
    try:
        cand = yaml.safe_load(s)
        if isinstance(cand, str) and ":" in cand:
            cand = yaml.safe_load(cand)
    except Exception:
        cand = None

    alias = {
        "logical_consistency": ["logical_consistency", "logic", "consistency", "logicalConsistency"],
        "specificity_and_detail": ["specificity_and_detail", "specificity", "detail", "specificityAndDetail"],
        "intuitive_depth": ["intuitive_depth", "intuition", "intuitiveDepth"],
        "clarity_and_conciseness": ["clarity_and_conciseness", "clarity", "conciseness", "clarityAndConciseness"],
    }

    def _normalize_dict(dct: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, al in alias.items():
            for a in al:
                if a in dct:
                    try:
                        out[k] = float(str(dct[a]).strip())
                        break
                    except Exception:
                        pass
        for k in list(out.keys()):
            out[k] = max(0.0, min(1.0, out[k]))
        if out:
            mean = sum(out.values()) / len(out)
            for k in alias.keys():
                out.setdefault(k, max(0.0, min(1.0, mean)))
        return out

    if isinstance(cand, dict):
        d = _normalize_dict(cand)
        if d:
            return d

    # 2) 行パース
    d2: dict[str, float] = {}
    for line in s.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        k = key.strip().lower().replace(" ", "").replace("-", "_")
        v = val.strip()
        mapping = {
            "logicalconsistency": "logical_consistency",
            "logical_consistency": "logical_consistency",
            "specificity_and_detail": "specificity_and_detail",
            "specificity": "specificity_and_detail",
            "detail": "specificity_and_detail",
            "intuitivedepth": "intuitive_depth",
            "intuitive_depth": "intuitive_depth",
            "clarity_and_conciseness": "clarity_and_conciseness",
            "clarity": "clarity_and_conciseness",
            "conciseness": "clarity_and_conciseness",
        }
        k = mapping.get(k, k)
        if k in {"logical_consistency", "specificity_and_detail", "intuitive_depth", "clarity_and_conciseness"}:
            try:
                d2[k] = max(0.0, min(1.0, float(v)))
            except Exception:
                pass
    if d2:
        if len(d2) < 4:
            mean = sum(d2.values()) / len(d2)
            for k in ("logical_consistency", "specificity_and_detail", "intuitive_depth", "clarity_and_conciseness"):
                d2.setdefault(k, max(0.0, min(1.0, mean)))
        return d2
    return {}


def _parse_target_agents_response(resp: str, allowed_names: list[str]) -> str:
    """対象エージェント応答の頑健パーサ。"""
    if not resp:
        return "null"
    s = _strip_code_fences(resp).strip()
    try:
        obj = yaml.safe_load(s)
    except Exception:
        obj = None

    names_set = set(allowed_names)

    if isinstance(obj, list):
        picked = [x for x in obj if isinstance(x, str) and x in names_set]
        return ",".join(picked) if picked else "null"
    if isinstance(obj, str):
        s2 = obj.strip()
        if s2.lower() in {"all", "全体"}:
            return "all"
        toks = [t.strip() for t in re.split(r"[,\s]+", s2) if t.strip()]
        picked = [t for t in toks if t in names_set]
        return ",".join(picked) if picked else "null"

    s3 = s.lower()
    if "all" in s3 or "全体" in s:
        return "all"
    toks = [t.strip() for t in re.split(r"[,\s]+", s) if t.strip()]
    picked = [t for t in toks if t in names_set]
    return ",".join(picked) if picked else "null"


def _extract_targets_by_names(text: str, allowed_names: list[str]) -> list[str]:
    """本文中から生存エージェント名（@Mention含む）を厳密抽出。"""
    if not text:
        return []
    names = set()
    for name in allowed_names:
        # @Name / Name のいずれも許容（単語境界をできるだけ尊重）
        if re.search(rf"@{re.escape(name)}\b", text):
            names.add(name)
        elif re.search(rf"\b{re.escape(name)}\b", text):
            names.add(name)
    return list(names)


def _contains_question(text: str) -> bool:
    if not text:
        return False
    s = text.strip()
    if "?" in s or "？" in s:
        return True
    # 簡易疑問語
    return bool(re.search(r"\b(who|what|why|how|when|where|can|could|would|should|do you|did you|will you)\b", s, re.I) or
                re.search(r"(ですか|ますか|でしょうか|誰|なぜ|どこ|いつ|どうして)", s))


def _has_role_word(text: str) -> bool:
    """本文に役職語（英/日）が出現しているか。"""
    if not text:
        return False
    return _RE_ROLE_WORD.search(text) is not None


class AnalysisTracker:
    """トーク分析を行い analysis.yml を生成するクラス."""

    def __init__(
        self,
        config: dict[str, Any],
        agent_name: str,
        game_id: str,
        agent_logger=None,
        agent_obj=None,
    ) -> None:
        self.config = config or {}
        self.agent_name = agent_name
        self.game_id = game_id
        self.packet_idx = 0
        self.agent_logger = agent_logger
        self.agent_obj = agent_obj

        self.analysis_history: dict[int, list[dict[str, Any]]] = {}
        self.seen_talk_keys: set[str] = set()
        self.last_analyzed_talk_count = 0

        # フォールバック LLM 初期化（config にプロンプトがある場合のみ使用）
        self.llm_model = None
        if bool(self.config.get("analysis", {}).get("enable_local_fallback_llm", True)):
            try:
                self._initialize_llm()
            except Exception:
                self.llm_model = None

        # 出力先
        self._setup_output_directory()
        self._ensure_dirs_and_touch_outputs()

    def _initialize_llm(self) -> None:
        """ローカルLLM（フォールバック）初期化."""
        load_dotenv(_resolve_project_root(Path(__file__).resolve()) / "config" / ".env")

        model_type = str(self.config.get("llm", {}).get("type", "openai"))
        print(f"[AnalysisTracker] Initializing fallback LLM with type: {model_type}")

        try:
            match model_type:
                case "openai":
                    api_key = os.environ.get("OPENAI_API_KEY", "")
                    if not api_key:
                        print("[AnalysisTracker] Warning: OPENAI_API_KEY not found")
                    self.llm_model = ChatOpenAI(
                        model=str(self.config.get("openai", {}).get("model", "gpt-4o")),
                        temperature=float(self.config.get("openai", {}).get("temperature", 0.3)),
                        api_key=SecretStr(api_key),
                    )
                case "google":
                    api_key = os.environ.get("GOOGLE_API_KEY", "")
                    if not api_key:
                        print("[AnalysisTracker] Warning: GOOGLE_API_KEY not found")
                    self.llm_model = ChatGoogleGenerativeAI(
                        model=str(self.config.get("google", {}).get("model", "gemini-2.0-flash-lite")),
                        temperature=float(self.config.get("google", {}).get("temperature", 0.3)),
                        google_api_key=SecretStr(api_key),
                    )
                case "ollama":
                    self.llm_model = ChatOllama(
                        model=str(self.config.get("ollama", {}).get("model", "llama3.1")),
                        temperature=float(self.config.get("ollama", {}).get("temperature", 0.3)),
                        base_url=str(self.config.get("ollama", {}).get("base_url", "http://localhost:11434")),
                    )
                case _:
                    print(f"[AnalysisTracker] Unknown LLM type: {model_type}")
                    self.llm_model = None

            if self.llm_model:
                print(f"[AnalysisTracker] Fallback LLM initialized: {type(self.llm_model).__name__}")
            else:
                print("[AnalysisTracker] Fallback LLM not initialized")

        except Exception as e:
            print(f"[AnalysisTracker] Failed to initialize fallback LLM: {e}")
            import traceback
            traceback.print_exc()
            self.llm_model = None

    def _setup_output_directory(self) -> None:
        """出力ディレクトリの設定."""
        project_root = _resolve_project_root(Path(__file__).resolve())
        self.output_dir = project_root / "info" / "bdi_info" / "micro_bdi" / self.game_id / self.agent_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_file = self.output_dir / "analysis.yml"

        print(f"[AnalysisTracker] analysis_file abs path = {self.analysis_file.resolve()}")
        print(f"[AnalysisTracker] game_id = {self.game_id}, agent_name = {self.agent_name}")

    def _talk_key(self, t: Talk) -> str:
        """トークの一意キー."""
        day = getattr(t, "day", -1)
        idx = getattr(t, "idx", None)
        if idx is not None:
            return f"{day}:{idx}"
        turn = getattr(t, "turn", -1)
        agent = getattr(t, "agent", "")
        h = hashlib.sha1(str(getattr(t, "text", "")).encode("utf-8", "ignore")).hexdigest()[:16]
        return f"{day}:{turn}:{agent}:{h}"

    def _is_meaningful_other_utterance(self, t: Talk) -> bool:
        """有意味な他者の発話か."""
        agent = getattr(t, "agent", "unknown")
        text = getattr(t, "text", "")
        txt = str(text).strip()
        if not text or txt == "":
            return False
        if txt in {"Skip", "Over"}:
            return False
        if getattr(t, "skip", False) or getattr(t, "over", False):
            return False
        if agent == self.agent_name:
            return False
        return True

    # ====== LLM ラッパ ======
    def _agent_llm_call(self, prompt_key: str, extra_vars: dict, log_tag: str) -> str | None:
        """agent_obj.center の LLM 呼び出し（失敗時 None）."""
        if self.agent_obj is None:
            return None
        try:
            resp = self.agent_obj.send_message_to_llm(prompt_key, extra_vars=extra_vars, log_tag=log_tag)
            return resp if isinstance(resp, str) else (None if resp is None else str(resp))
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.llm_error(log_tag, str(e))
            return None

    def _local_llm_call(self, prompt_key: str, extra_vars: dict, log_tag: str) -> str | None:
        """フォールバック LLM 呼び出し（config にプロンプトが無ければ実行しない）."""
        # config に該当プロンプトがない場合は使わない（コードにプロンプトを持たない方針）
        if not (self.config.get("prompt", {}) or {}).get(prompt_key):
            return None
        if self.llm_model is None:
            return None
        from jinja2 import Template
        from langchain_core.messages import HumanMessage
        from langchain_core.output_parsers import StrOutputParser

        prompt_tpl = self.config["prompt"][prompt_key]
        prompt = Template(prompt_tpl).render(**extra_vars)

        messages = [HumanMessage(content=prompt)]
        try:
            resp = (self.llm_model | StrOutputParser()).invoke(messages)
            if self.agent_logger:
                self.agent_logger.llm_interaction(log_tag, prompt, resp, type(self.llm_model).__name__)
            return resp
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.llm_error(log_tag, str(e), prompt)
            return None

    # ====== 一件の発話解析 ======
    def analyze_talk(
        self,
        talk_history: list[Talk],
        info: Info,
        request_count: int | None = None,
        **_: Any,
    ) -> int:
        """トーク履歴を分析して analysis.yml を更新。"""
        if not talk_history:
            print("[AnalysisTracker] analyze_talk: talk_history is empty/None")
            return 0

        print(f"[AnalysisTracker] Analyzing talk for {self.agent_name}")
        print(f"[AnalysisTracker] Total talk_history: {len(talk_history)}")
        print(f"[AnalysisTracker] Seen talk keys count: {len(self.seen_talk_keys)}")

        cfg = self._get_fixture_config()
        if cfg["enable"]:
            print(f"[AnalysisTracker] Fixture mode ENABLED: output={cfg['output_file']}, max={cfg['max_per_call']}, apply_to={cfg['apply_to_agents']}")

        self.packet_idx += 1
        if self.packet_idx not in self.analysis_history:
            self.analysis_history[self.packet_idx] = []

        added_entries = 0
        candidates: list[tuple[str, str, Any, Any]] = []

        for t in talk_history:
            k = self._talk_key(t)
            if k in self.seen_talk_keys:
                continue

            is_self = (getattr(t, "agent", "") == self.agent_name)
            apply_ok = (cfg["apply_to_agents"] == "all") or (not is_self)
            txt = str(getattr(t, "text", "") or "").strip()
            is_target_text = txt in set(cfg["rewrite_targets"])
            is_flagged = bool(getattr(t, "skip", False) or getattr(t, "over", False))

            analyzable = False

            if cfg["enable"] and apply_ok and (is_target_text or is_flagged):
                from types import SimpleNamespace
                repl = self._pick_fixture_text(t, cfg)
                if repl:
                    proxy = SimpleNamespace(
                        idx=getattr(t, "idx", None),
                        day=getattr(t, "day", -1),
                        turn=getattr(t, "turn", -1),
                        agent=getattr(t, "agent", ""),
                        text=repl,
                        skip=False,
                        over=False,
                    )
                    candidates.append((k, "fixture", proxy, t))
                    print(f"[AnalysisTracker] Fixture replacement: '{txt}' -> '{repl}' for {getattr(t, 'agent', 'unknown')}")
                    analyzable = True
            elif self._is_meaningful_other_utterance(t):
                candidates.append((k, "real", t, None))
                analyzable = True

            if not analyzable:
                self.seen_talk_keys.add(k)

        if not candidates:
            print("[AnalysisTracker] No analyzable talks in this call")
            self.last_analyzed_talk_count = len(talk_history)
            if cfg["enable"] and cfg.get("trace_file"):
                self._record_heartbeat_trace(cfg, "empty_candidates")
            return 0

        process_count = min(len(candidates), cfg["max_per_call"] if cfg["enable"] else len(candidates))
        print(f"[AnalysisTracker] Found {len(candidates)} candidates, processing {process_count}")

        processed_keys = []
        for i, (key, kind, talk, origin) in enumerate(candidates[:process_count]):
            agent_name = getattr(talk, "agent", "unknown")
            text = getattr(talk, "text", "")
            print(f"[AnalysisTracker] Analyzing ({kind}) talk {i+1}/{process_count}: {agent_name} - {text}")

            pkt, pos = self._insert_stub_entry_before_llm(talk, kind)

            analysis_entry = self._analyze_talk_entry(talk, info)
            if not analysis_entry:
                analysis_entry = self._fallback_entry_for(talk)
                print(f"[AnalysisTracker] No analysis entry; wrote fallback for: {text}")
            else:
                print(f"[AnalysisTracker] Analysis entry created ({kind}): {analysis_entry['type']} from {analysis_entry['from']}")

            if kind == "fixture" and cfg.get("trace_file") and origin is not None:
                self._record_fixture_trace(origin, talk, cfg)

            try:
                self._replace_stub_with_result(pkt, pos, analysis_entry)
            except Exception as e:
                print(f"[AnalysisTracker] Failed to replace stub at pkt={pkt}, pos={pos}: {e}")
                self.analysis_history[pkt].append(analysis_entry)

            try:
                self.save_analysis(run_downstream=False)
            except Exception as e:
                print(f"[AnalysisTracker] Per-talk save failed: {e}")
                if self.agent_logger:
                    self.agent_logger.logger.exception(f"Per-talk save failed: {e}")

            processed_keys.append(key)
            added_entries += 1

        for key in processed_keys:
            self.seen_talk_keys.add(key)

        if len(candidates) > process_count:
            print(f"[AnalysisTracker] {len(candidates) - process_count} candidates deferred to next call")

        self.last_analyzed_talk_count = len(talk_history)
        print(f"[AnalysisTracker] newly added entries in this call: {added_entries}")

        try:
            print(f"[AnalysisTracker] === save_analysis 開始 (final) ===")
            self.save_analysis(run_downstream=True)
            if cfg["enable"] and cfg.get("trace_file"):
                self.save_fixture_trace()
            print(f"[AnalysisTracker] === save_analysis 終了 (final) ===")
        except Exception as e:
            print(f"[AnalysisTracker] Final save failed: {e}")
            if self.agent_logger:
                self.agent_logger.logger.exception(f"Final save failed: {e}")

        return added_entries

    def _fallback_entry_for(self, talk: Any) -> dict[str, Any]:
        """LLM失敗時などのフォールバックエントリ."""
        return {
            "content": getattr(talk, "text", ""),
            "type": "null",
            "from": getattr(talk, "agent", "unknown"),
            "to": "null",
            "credibility": 0.0,
        }

    def _analyze_talk_entry(self, talk: Talk, info: Info) -> dict[str, Any] | None:
        """個別のトーク発話を分析."""
        if not talk.text or talk.text.strip() == "":
            print(f"[AnalysisTracker] Skipping empty talk from {talk.agent}")
            return None

        if talk.text.strip() == "Over":
            print(f"[AnalysisTracker] Processing 'Over' talk from {talk.agent}")
            return {
                "content": talk.text,
                "type": "null",
                "from": talk.agent,
                "to": "null",
                "credibility": 0.0,
            }

        content = talk.text
        from_agent = talk.agent

        print(f"[AnalysisTracker] Processing talk from {from_agent}: '{content}'")
        print(f"[AnalysisTracker] Fallback LLM available: {self.llm_model is not None}")

        # 1) LLM 一次判定
        names = list(info.status_map.keys())
        extra = {"content": content, "agent_names": names}

        mt_resp = self._agent_llm_call("analyze_message_type", extra, "message_type_analysis")
        if not mt_resp:
            mt_resp = self._local_llm_call("analyze_message_type", extra, "message_type_analysis")
        initial_type = self._parse_type_token(mt_resp)

        # target_agents には msg_type/speaker ヒントも渡す（プロンプト側で利用）
        extra_to = {"content": content, "agent_names": names, "msg_type": initial_type, "speaker": from_agent}
        ta_resp = self._agent_llm_call("analyze_target_agents", extra_to, "target_agents_analysis")
        if not ta_resp:
            ta_resp = self._local_llm_call("analyze_target_agents", extra_to, "target_agents_analysis")
        initial_to = _parse_target_agents_response(ta_resp or "", names)
        if initial_to == "null":
            # LLMが取れない時の補完
            guessed = _extract_targets_by_names(content, names)
            if guessed:
                initial_to = ",".join(guessed)

        # 2) ルール後処理（役職語フィルタ + 厳密仕様の適用）
        final_type, final_to = self._postprocess_type(content, initial_type, initial_to, names, from_agent)

        # 3) 信憑性
        credibility = self._analyze_credibility(content, info, from_agent)

        print(f"[AnalysisTracker] Analysis results - type: {final_type}, to: {final_to}, credibility: {credibility}")

        entry = {
            "content": content,
            "type": final_type,
            "from": from_agent,
            "to": final_to,
            "credibility": credibility,
        }
        if os.environ.get("ANALYSIS_SAVE_CREDIBILITY_DEBUG", "0").lower() in {"1", "true", "on"}:
            dbg = getattr(self, "_last_credibility_debug", None)
            if dbg:
                entry["credibility_raw"] = float(dbg.get("avg_raw", 0.0))
                entry["credibility_breakdown"] = dbg.get("raw", {})
        return entry

    def _parse_type_token(self, resp: str | None) -> str:
        """LLMからのtypeトークンを一語で抽出。"""
        if not resp:
            return "null"
        s = _strip_code_fences(str(resp)).strip().lower()
        tokens = re.split(r"[\s,|/]+", s)
        allowed = ["co", "question", "positive", "negative", "null"]
        for t in tokens:
            if t in allowed:
                return t
        for t in allowed:
            if t in s:
                return t
        return "null"

    def _postprocess_type(
        self,
        content: str,
        mtype: str,
        to_agents: str,
        allowed_names: list[str],
        speaker: str,
    ) -> tuple[str, str]:
        """
        後処理仕様:
          - co は LLM 一次判定を尊重する。ただし「本文に役職語が1つも無い場合」は co 不許可（降格）。
            * 役職語ありで co の場合、to が空/全体なら最小限の補完（本文の名前→なければ speaker）。
          - negative/positive: 必ず特定宛先（to!=all/null）。満たせないと null。
          - question: to が空なら本文から対象抽出 / 無ければ to=all。
        """
        s = content or ""
        role_present = _has_role_word(s)

        # ① co の扱い（役職語フィルタ）
        if mtype == "co":
            if not role_present:
                # 役職語がないcoは不許可 → 簡易再分類
                mtype = self._fallback_non_co_type_guess(s, allowed_names)
            else:
                # LLMのco判定を採用。toが明示されていない場合のみ最小限で補完。
                if to_agents in (None, "", "null", "all"):
                    guessed = _extract_targets_by_names(s, allowed_names)
                    to_agents = ",".join(guessed) if guessed else speaker
                return "co", to_agents

        # ② positive / negative は必ず特定宛先
        if mtype in ("positive", "negative"):
            if to_agents in (None, "", "null", "all"):
                guessed = _extract_targets_by_names(s, allowed_names)
                if guessed:
                    to_agents = ",".join(guessed)
                else:
                    return "null", "null"
            return mtype, to_agents

        # ③ question は to 補完
        if mtype == "question" or _contains_question(s):
            if to_agents in (None, "", "null"):
                guessed = _extract_targets_by_names(s, allowed_names)
                to_agents = ",".join(guessed) if guessed else "all"
            return "question", to_agents

        # ④ どれでもない → null
        return "null", (to_agents if to_agents not in (None, "") else "null")

    def _fallback_non_co_type_guess(self, text: str, allowed_names: list[str]) -> str:
        """co降格後の簡易再分類（疑問符/否定/擁護語から推定）。"""
        s = text or ""
        has_neg = any(h in s for h in NEGATIVE_HINTS)
        has_pos = any(h in s for h in POSITIVE_HINTS)
        has_q = _contains_question(s)
        targets = _extract_targets_by_names(s, allowed_names)

        if has_q:
            return "question"
        if has_neg and targets:
            return "negative"
        if has_pos and targets:
            return "positive"
        return "null"

    # ====== 信憑性 ======
    def _analyze_credibility(self, content: str, info: Info, from_agent: str) -> float:
        raw_scores = self._get_credibility_scores(content, info)
        if not raw_scores:
            self._last_credibility_debug = {"raw": {}, "weights": {}, "avg_raw": 0.0, "avg_weighted": 0.0}
            return 0.0

        avg_raw = round(sum(raw_scores.values()) / len(raw_scores), 2)
        weights = self._get_statement_bias_weights(from_agent)

        if not weights or sum(weights.values()) == 0.0:
            avg_weighted = avg_raw
        else:
            ws = sum(weights.values())
            avg_weighted = round(
                sum(raw_scores[k] * (weights.get(k, 0.0) / ws) for k in raw_scores.keys()), 2
            )

        self._last_credibility_debug = {
            "raw": raw_scores,
            "weights": weights,
            "avg_raw": avg_raw,
            "avg_weighted": avg_weighted,
        }
        return avg_weighted

    def _get_credibility_scores(self, content: str, info: Info) -> dict[str, float]:
        extra = {"content": content, "agent_names": list(info.status_map.keys())}

        resp = self._agent_llm_call("analyze_credibility", extra, "credibility_analysis")
        scores = _parse_credibility_scores(resp or "")
        if scores:
            return scores

        resp2 = self._local_llm_call("analyze_credibility", extra, "credibility_analysis")
        scores2 = _parse_credibility_scores(resp2 or "")
        if scores2:
            return scores2

        # 軽量ヒューリスティック（最終手段）
        txt = content or ""
        has_mention = bool(re.search(r"@\w+|\b[A-Z][a-z]{2,}\b", txt))
        has_reason = bool(re.search(r"\b(because|since|why|reason)\b", txt, flags=re.I))
        length = len(txt)
        base = 0.5
        return {
            "logical_consistency": max(0.0, min(1.0, base + (0.08 if has_reason else -0.05))),
            "specificity_and_detail": max(0.0, min(1.0, base + (0.06 if has_mention else -0.04))),
            "intuitive_depth": 0.5,
            "clarity_and_conciseness": max(0.0, min(1.0, 0.56 if 20 <= length <= 160 else 0.46)),
        }

    def _get_statement_bias_weights(self, agent_name: str) -> dict[str, float]:
        """macro_belief.yml の statement_bias をプロジェクトルート解決で取得。"""
        try:
            root = _resolve_project_root(Path(__file__).resolve())
            path = root / "info" / "bdi_info" / "macro_bdi" / self.game_id / agent_name / "macro_belief.yml"
            if not path.exists():
                return {}
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            sb = (data.get("macro_belief", {})
                      .get("cognitive_bias", {})
                      .get("statement_bias", {})) or {}
            out = {}
            for k in ("logical_consistency", "specificity_and_detail", "intuitive_depth", "clarity_and_conciseness"):
                try:
                    v = float(sb.get(k, 0.0))
                except Exception:
                    v = 0.0
                out[k] = max(0.0, min(1.0, v))
            return out
        except Exception:
            return {}

    # ====== 保存系 ======
    def save_analysis(self, run_downstream: bool = True) -> None:
        """analysis.yml に保存（下流処理はフラグで制御）。"""
        cfg = self._get_fixture_config()
        target_path = self.output_dir / (cfg["output_file"] if cfg["enable"] else "analysis.yml")

        print(f"[AnalysisTracker] === save_analysis 開始 ===")
        print(f"[AnalysisTracker] agent_name: {self.agent_name}")
        print(f"[AnalysisTracker] game_id: {self.game_id}")
        print(f"[AnalysisTracker] Fixture mode: {'ENABLED' if cfg['enable'] else 'DISABLED'}")
        print(f"[AnalysisTracker] Target file: {target_path.name}")
        print(f"[AnalysisTracker] analysis_history keys: {list(self.analysis_history.keys())}")

        all_entries = []
        for packet_idx in sorted(self.analysis_history.keys()):
            entries = self.analysis_history[packet_idx] or []
            print(f"[AnalysisTracker]   packet_idx {packet_idx}: {len(entries)} entries")
            all_entries.extend(entries)

        print(f"[AnalysisTracker] 合計エントリ数(含pending): {len(all_entries)}")
        all_entries = [e for e in all_entries if e.get("type") != "pending_analysis"]
        print(f"[AnalysisTracker] 合計エントリ数(出力対象): {len(all_entries)}")

        if all_entries:
            data = {
                i + 1: {
                    "content": e.get("content", ""),
                    "type": e.get("type", "null"),
                    "from": e.get("from", "unknown"),
                    "to": e.get("to", "null"),
                    "credibility": float(e.get("credibility", 0.0)),
                    **(
                        {
                            "credibility_raw": float(e.get("credibility_raw", 0.0)),
                            "credibility_breakdown": e.get("credibility_breakdown", {}),
                        }
                        if os.environ.get("ANALYSIS_SAVE_CREDIBILITY_DEBUG", "0").lower() in {"1", "true", "on"}
                        else {}
                    ),
                }
                for i, e in enumerate(all_entries)
            }
        else:
            data = {}

        try:
            with NamedTemporaryFile(
                "w", encoding="utf-8", delete=False, dir=str(self.output_dir),
                prefix="analysis_", suffix=".tmp"
            ) as tmp:
                yaml.safe_dump(data, tmp, allow_unicode=True, sort_keys=False, default_flow_style=False)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(str(tmp_path), str(target_path))
        except Exception as e:
            print(f"[AnalysisTracker] ERROR during file save: {e}")
            import traceback
            traceback.print_exc()
            if self.agent_logger:
                self.agent_logger.logger.error(f"Failed to save analysis file: {e}")
            raise

        file_size = target_path.stat().st_size
        print(f"[AnalysisTracker] saved: {target_path.resolve()} size={file_size}")
        print(f"[AnalysisTracker] 保存したデータキー数: {len(data)}")

        try:
            with open(target_path, "rb") as rf:
                blob = rf.read()
            print(f"[AnalysisTracker] readback right after save: {len(blob)} bytes, head={blob[:120]!r}")
            self._verify_saved_file(target_path, len(data))
        except Exception as e:
            print(f"[AnalysisTracker] readback failed: {e}")

        ds_flags = self._get_downstream_flags()

        if cfg["enable"] and "ANALYSIS_UPDATE_SELECT_SENTENCE" not in os.environ:
            ds_flags["update_select_sentence"] = False
        if cfg["enable"] and "ANALYSIS_UPDATE_INTENTION" not in os.environ:
            ds_flags["update_intention"] = False

        before = target_path.stat().st_size

        if not run_downstream:
            print("[AnalysisTracker] Downstream updates are skipped (run_downstream=False)")
            print("[AnalysisTracker] === save_analysis 終了 ===")
            return

        if ds_flags.get("update_select_sentence", True):
            print("[AnalysisTracker] Downstream(select_sentence): RUN")
            self._update_select_sentence()
            try:
                after1 = target_path.stat().st_size
                if after1 < before:
                    if self.agent_logger:
                        self.agent_logger.logger.warning(f"File size decreased after select_sentence: {before} -> {after1}")
                    print(f"[AnalysisTracker] WARNING: file size decreased after select_sentence: {before} -> {after1}")
            except Exception as e:
                print(f"[AnalysisTracker] stat after select_sentence failed: {e}")
        else:
            reason = "(disabled by fixture)" if cfg["enable"] else "(disabled by flag)"
            print(f"[AnalysisTracker] Downstream(select_sentence): SKIP {reason}")

        if ds_flags.get("update_intention", True):
            print("[AnalysisTracker] Downstream(intention): RUN")
            self._update_intention()
            try:
                after2 = target_path.stat().st_size
                if after2 < before:
                    if self.agent_logger:
                        self.agent_logger.logger.warning(f"File size decreased after intention: {after2} -> {before}")
                    print(f"[AnalysisTracker] WARNING: file size decreased after intention: {after2} -> {before}")
            except Exception as e:
                print(f"[AnalysisTracker] stat after intention failed: {e}")
        else:
            reason = "(disabled by fixture)" if cfg["enable"] else "(disabled by flag)"
            print(f"[AnalysisTracker] Downstream(intention): SKIP {reason}")

        print("[AnalysisTracker] === save_analysis 終了 ===")

    def save_fixture_trace(self) -> None:
        """Fixtureトレースファイル保存（冪等性保証）."""
        cfg = self._get_fixture_config()
        if not cfg["enable"] or not cfg.get("trace_file"):
            return

        trace_path = self.output_dir / cfg["trace_file"]

        print("[AnalysisTracker] === save_fixture_trace 開始 ===")

        existing_traces = {}
        if trace_path.exists():
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    existing_traces = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[AnalysisTracker] Failed to read existing trace: {e}")

        try:
            with NamedTemporaryFile(
                "w", encoding="utf-8", delete=False, dir=str(self.output_dir),
                prefix="trace_", suffix=".tmp"
            ) as tmp:
                if not existing_traces:
                    tmp.write("{}\n")
                else:
                    yaml.safe_dump(existing_traces, tmp, allow_unicode=True, sort_keys=False, default_flow_style=False)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(str(tmp_path), str(trace_path))
        except Exception as e:
            print(f"[AnalysisTracker] ERROR during trace save: {e}")
            if self.agent_logger:
                self.agent_logger.logger.error(f"Failed to save trace file: {e}")
            raise

        print(f"[AnalysisTracker] saved trace: {trace_path.resolve()}")

        try:
            trace_size = trace_path.stat().st_size
            print(f"[AnalysisTracker] trace file size: {trace_size} bytes")
            self._verify_saved_file(trace_path, len(existing_traces))
        except Exception as e:
            print(f"[AnalysisTracker] trace file verification failed: {e}")

        print("[AnalysisTracker] === save_fixture_trace 終了 ===")

    def force_save_with_status(self) -> None:
        """強制保存（デバッグ用）."""
        print("[AnalysisTracker] Force saving analysis status:")
        print(f"  - agent_name: {self.agent_name}")
        print(f"  - game_id: {self.game_id}")
        print(f"  - packet_idx: {self.packet_idx}")
        print(f"  - last_analyzed_talk_count: {self.last_analyzed_talk_count}")
        print(f"  - Fallback LLM available: {self.llm_model is not None}")
        print(f"  - analysis_history keys: {list(self.analysis_history.keys())}")
        self.save_analysis(run_downstream=True)

    def _update_select_sentence(self) -> None:
        try:
            from .select_sentence import SelectSentenceTracker
            tracker = SelectSentenceTracker(self.agent_name, self.game_id)
            tracker.process_select_sentence()
        except ImportError:
            pass
        except Exception as e:
            print(f"Failed to update select_sentence: {e}")

    def _update_intention(self) -> None:
        try:
            from .intention_tracker import IntentionTracker
            tracker = IntentionTracker(self.config, self.agent_name, self.game_id)
            tracker.process_intention()
        except ImportError:
            pass
        except Exception as e:
            print(f"Failed to update intention: {e}")

    # -------- Fixture mode helpers --------
    def _env_bool(self, name: str, default: bool = False) -> bool:
        val = os.environ.get(name, "").lower()
        if val in {"1", "true", "yes", "on"}:
            return True
        elif val in {"0", "false", "no", "off"}:
            return False
        return default

    def _env_list(self, name: str, sep: str = ",", default: list | None = None) -> list[str]:
        val = os.environ.get(name, "")
        if not val:
            return default or []
        return [s.strip() for s in val.split(sep) if s.strip()]

    def _get_fixture_config(self) -> dict[str, Any]:
        cfg_base = self.config.get("analysis", {}).get("fixture_mode", {}) if self.config else {}
        cfg = {
            "enable": cfg_base.get("enable", False),
            "output_file": cfg_base.get("output_file", "analysis_test.yml"),
            "trace_file": cfg_base.get("trace_file", "analysis_fixture_trace.yml"),
            "rewrite_targets": cfg_base.get("rewrite_targets", ["Skip", "Over"]),
            "max_per_call": cfg_base.get("max_per_call", 999),
            "apply_to_agents": cfg_base.get("apply_to_agents", "others"),
            "utterances": {
                "default": cfg_base.get("utterances", {}).get("default", DEFAULT_FIXTURE_UTTERANCES[:]),
                "by_agent": cfg_base.get("utterances", {}).get("by_agent", {}),
            },
        }

        if "ANALYSIS_FIXTURE_ENABLE" in os.environ:
            cfg["enable"] = self._env_bool("ANALYSIS_FIXTURE_ENABLE")
        if "ANALYSIS_FIXTURE_OUTPUT_FILE" in os.environ:
            cfg["output_file"] = os.environ["ANALYSIS_FIXTURE_OUTPUT_FILE"]
        if "ANALYSIS_FIXTURE_TRACE_FILE" in os.environ:
            cfg["trace_file"] = os.environ["ANALYSIS_FIXTURE_TRACE_FILE"]
        if "ANALYSIS_FIXTURE_TARGETS" in os.environ:
            cfg["rewrite_targets"] = self._env_list("ANALYSIS_FIXTURE_TARGETS")
        if "ANALYSIS_FIXTURE_MAX_PER_CALL" in os.environ:
            try:
                cfg["max_per_call"] = int(os.environ["ANALYSIS_FIXTURE_MAX_PER_CALL"])
            except ValueError:
                pass
        if "ANALYSIS_FIXTURE_APPLY_TO" in os.environ:
            cfg["apply_to_agents"] = os.environ["ANALYSIS_FIXTURE_APPLY_TO"]
        if "ANALYSIS_FIXTURE_UTTERANCES_DEFAULT" in os.environ:
            cfg["utterances"]["default"] = self._env_list("ANALYSIS_FIXTURE_UTTERANCES_DEFAULT", sep="|")

        return cfg

    def _pick_fixture_text(self, talk: Any, cfg: dict[str, Any]) -> str | None:
        agent = getattr(talk, "agent", "")
        candidates = cfg["utterances"].get("by_agent", {}).get(agent, [])
        if not candidates:
            candidates = cfg["utterances"].get("default", [])
        if not candidates:
            return None
        day = getattr(talk, "day", 0)
        turn = getattr(talk, "turn", 0)
        idx = getattr(talk, "idx", 0) or 0
        hash_val = (day * 1000 + turn * 10 + idx) % len(candidates)
        return candidates[hash_val]

    def _record_fixture_trace(self, origin: Any, proxy: Any, cfg: dict[str, Any]) -> None:
        trace_file = cfg.get("trace_file")
        if not trace_file:
            return
        trace_path = self.output_dir / trace_file

        day = getattr(origin, "day", -1)
        idx = getattr(origin, "idx", None)
        turn = getattr(origin, "turn", -1)
        agent = getattr(origin, "agent", "unknown")
        original_text = getattr(origin, "text", "")
        replaced_text = getattr(proxy, "text", "")

        text_hash = hashlib.sha1(str(original_text).encode("utf-8", "ignore")).hexdigest()[:8]
        key = f"{day}:{idx}:{text_hash}" if idx is not None else f"{day}:{turn}:{text_hash}"

        trace_entry = {
            key: {
                "from_agent": agent,
                "original": original_text,
                "replaced": replaced_text,
                "timestamp": datetime.now().isoformat(),
            }
        }

        existing_traces = {}
        if trace_path.exists():
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    existing_traces = yaml.safe_load(f) or {}
            except Exception:
                pass

        existing_traces.update(trace_entry)

        try:
            with NamedTemporaryFile(
                "w", encoding="utf-8", delete=False, dir=str(self.output_dir),
                prefix="trace_", suffix=".tmp"
            ) as tmp:
                yaml.safe_dump(existing_traces, tmp, allow_unicode=True, sort_keys=False, default_flow_style=False)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(str(tmp_path), str(trace_path))
            print(f"[AnalysisTracker] Fixture trace recorded: {key}")
        except Exception as e:
            print(f"[AnalysisTracker] Failed to record fixture trace: {e}")

    def _get_downstream_flags(self) -> dict[str, bool]:
        flags = {"update_select_sentence": True, "update_intention": True}
        if "ANALYSIS_UPDATE_SELECT_SENTENCE" in os.environ:
            flags["update_select_sentence"] = self._env_bool("ANALYSIS_UPDATE_SELECT_SENTENCE", True)
        if "ANALYSIS_UPDATE_INTENTION" in os.environ:
            flags["update_intention"] = self._env_bool("ANALYSIS_UPDATE_INTENTION", True)
        return flags

    def _ensure_dirs_and_touch_outputs(self) -> None:
        cfg = self._get_fixture_config()
        paths = []
        trace_path = None
        if cfg["enable"]:
            paths.append(self.output_dir / cfg["output_file"])
            if cfg.get("trace_file"):
                trace_path = self.output_dir / cfg["trace_file"]
                paths.append(trace_path)
        else:
            paths.append(self.analysis_file)

        for path in paths:
            if not path.exists():
                try:
                    if path == trace_path:
                        self._create_trace_with_heartbeat(path)
                    else:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write("{}\n")
                    print(f"[AnalysisTracker] Pre-touched: {path}")
                except Exception as e:
                    print(f"[AnalysisTracker] Failed to pre-touch {path}: {e}")

    def _create_trace_with_heartbeat(self, trace_path: Path) -> None:
        meta_key = f"meta:tracker_init:{int(time.time())}:{random.randint(1000, 9999)}"
        heartbeat_entry = {
            meta_key: {
                "from_agent": None,
                "original": "",
                "replaced": "",
                "timestamp": datetime.now().isoformat(),
                "reason": "tracker_init",
            }
        }
        try:
            with open(trace_path, "w", encoding="utf-8") as f:
                yaml.dump(heartbeat_entry, f, default_flow_style=False, allow_unicode=True)
            print(f"[AnalysisTracker] Created trace with heartbeat: {trace_path}")
        except Exception as e:
            print(f"[AnalysisTracker] Failed to create heartbeat trace {trace_path}: {e}")
            with open(trace_path, "w", encoding="utf-8") as f:
                f.write("{}\n")

    def _record_heartbeat_trace(self, cfg: dict[str, Any], reason: str) -> None:
        trace_file = cfg.get("trace_file")
        if not trace_file:
            return
        trace_path = self.output_dir / trace_file

        meta_key = f"meta:heartbeat:{reason}:{int(time.time())}:{random.randint(1000, 9999)}"
        heartbeat_entry = {
            "from_agent": None,
            "original": "",
            "replaced": "",
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
        }

        existing_traces = {}
        if trace_path.exists():
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    existing_traces = yaml.safe_load(f) or {}
            except Exception:
                pass

        existing_traces[meta_key] = heartbeat_entry

        try:
            with NamedTemporaryFile(
                "w", encoding="utf-8", delete=False, dir=str(self.output_dir),
                prefix="heartbeat_", suffix=".tmp"
            ) as tmp_file:
                yaml.dump(existing_traces, tmp_file, default_flow_style=False, allow_unicode=True)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_file.name, trace_path)
            print(f"[AnalysisTracker] Recorded heartbeat trace: {reason}")
        except Exception as e:
            print(f"[AnalysisTracker] Failed to record heartbeat trace: {e}")
            try:
                if 'tmp_file' in locals() and os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
            except Exception:
                pass

    def _insert_stub_entry_before_llm(self, talk: Talk, kind: str) -> tuple[int, int]:
        pkt = self.packet_idx
        lst = self.analysis_history.setdefault(pkt, [])
        stub_entry = {
            "content": f"[PENDING] Analyzing: {str(getattr(talk, 'text', ''))[:50]}...",
            "type": "pending_analysis",
            "from": getattr(talk, "agent", "unknown"),
            "to": "all",
            "credibility": 0.0,
            "timestamp": datetime.now().isoformat(),
            "status": "pending_llm_analysis",
            "original_text": getattr(talk, "text", ""),
            "analysis_kind": kind,
        }
        lst.append(stub_entry)
        pos = len(lst) - 1

        cfg = self._get_fixture_config()
        if cfg["enable"] and cfg.get("trace_file"):
            self._record_heartbeat_trace(cfg, f"stub_inserted_{kind}")

        print(f"[AnalysisTracker] Inserted in-memory stub at pkt={pkt}, pos={pos}")
        return pkt, pos

    def _replace_stub_with_result(self, pkt: int, pos: int, result: dict[str, Any]) -> None:
        if pkt not in self.analysis_history:
            raise KeyError(f"packet {pkt} not found")
        if pos < 0 or pos >= len(self.analysis_history[pkt]):
            raise IndexError(f"position {pos} out of range for packet {pkt}")
        self.analysis_history[pkt][pos] = result
        print(f"[AnalysisTracker] Replaced stub at pkt={pkt}, pos={pos}")

    def _verify_saved_file(self, file_path: Path, expected_entry_count: int) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"Saved file does not exist: {file_path}")
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Saved file is empty: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Saved file is not a dict: {type(loaded)}")
        actual = len(loaded)
        if actual != expected_entry_count:
            print(f"[AnalysisTracker] WARNING: Entry count mismatch. Expected: {expected_entry_count}, Actual: {actual}")
        print(f"[AnalysisTracker] File verification passed: {file_path.name} ({file_size} bytes, {actual} entries)")

    def sync_with_trace(self) -> None:
        cfg = self._get_fixture_config()
        if not cfg["enable"] or not cfg.get("trace_file"):
            print("[AnalysisTracker] sync_with_trace: Fixture mode disabled, skipping")
            return

        trace_path = self.output_dir / cfg["trace_file"]
        analysis_path = self.output_dir / cfg["output_file"]

        print("[AnalysisTracker] === sync_with_trace 開始 ===")
        print(f"[AnalysisTracker] Trace file: {trace_path}")
        print(f"[AnalysisTracker] Analysis file: {analysis_path}")

        existing_traces = {}
        if trace_path.exists():
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    existing_traces = yaml.safe_load(f) or {}
                print(f"[AnalysisTracker] Loaded {len(existing_traces)} trace entries")
            except Exception as e:
                print(f"[AnalysisTracker] Failed to load trace file: {e}")
                return

        existing_analysis = {}
        if analysis_path.exists():
            try:
                with open(analysis_path, "r", encoding="utf-8") as f:
                    existing_analysis = yaml.safe_load(f) or {}
                print(f"[AnalysisTracker] Loaded {len(existing_analysis)} analysis entries")
            except Exception as e:
                print(f"[AnalysisTracker] Failed to load analysis file: {e}")

        sync_needed = False

        all_current_entries = []
        for packet_idx in sorted(self.analysis_history.keys()):
            entries = self.analysis_history[packet_idx] or []
            all_current_entries.extend([e for e in entries if e.get("type") != "pending_analysis"])

        current_count = len(all_current_entries)
        file_count = len(existing_analysis)
        if current_count != file_count:
            print(f"[AnalysisTracker] Count mismatch detected: memory={current_count}, file={file_count}")
            sync_needed = True

        if sync_needed or len(existing_traces) == 0:
            self._record_heartbeat_trace(cfg, "sync_executed")
            print("[AnalysisTracker] Added sync heartbeat to trace")

        if sync_needed:
            print("[AnalysisTracker] Executing file sync...")
            try:
                self.save_analysis(run_downstream=False)
                self.save_fixture_trace()
                print("[AnalysisTracker] File sync completed")
            except Exception as e:
                print(f"[AnalysisTracker] File sync failed: {e}")
        else:
            print("[AnalysisTracker] Files already in sync")

        print("[AnalysisTracker] === sync_with_trace 終了 ===")
