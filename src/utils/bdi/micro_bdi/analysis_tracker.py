"""analysis.ymlを生成するためのモジュール."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
import hashlib
import time
import random

import yaml
import ulid

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Talk

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import SecretStr
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

# Default prompts for analysis (fallback when config.yml doesn't have them)
DEFAULT_PROMPTS = SimpleNamespace(
    analyze_message_type="""You are classifying a werewolf-game utterance.
Utterance: {{ content }}
Agents: {{ agent_names|join(", ") }}
Return exactly one of: co, question, positive, negative, null""",
    
    analyze_target_agents="""Identify the target agents mentioned or addressed.
Utterance: {{ content }}
Agents: {{ agent_names|join(", ") }}
Output EXACTLY ONE of:
- comma-separated agent names (must be from the list)
- all
- null""",
    
    analyze_credibility="""Score the utterance on four 0-1 metrics.
Utterance: {{ content }}
Agents: {{ agent_names|join(", ") }}
Return EXACTLY this format:
logical_consistency: 0.XX
specificity_and_detail: 0.XX
intuitive_depth: 0.XX
clarity_and_conciseness: 0.XX"""
)

# Default fixture utterances (when neither config nor env var provides them)
DEFAULT_FIXTURE_UTTERANCES = [
    "Are there any seer claims?",
    "Please give a tentative vote with a reason.",
    "Share one town and one wolf read.",
    "Who do you think is suspicious and why?",
    "What is your current analysis of the situation?"
]


def _resolve_project_root(start: Path) -> Path:
    """プロジェクトルートを解決する.
    
    config/.envファイルを手掛かりに最上位ディレクトリを探索する。
    見つからない場合は、startから4階層上をフォールバックとして使用。
    """
    for p in [start] + list(start.parents):
        if (p / "config" / ".env").exists():
            return p
    # フォールバック
    parents = list(start.parents)
    return parents[3] if len(parents) >= 4 else (parents[-1] if parents else start)


def _safe_game_timestamp_from_ulid(game_id: str) -> str:
    """ULIDからゲームタイムスタンプを安全に生成する.
    
    ULID解析に失敗した場合は現在時刻をフォールバックとして使用。
    """
    try:
        u: ulid.ULID = ulid.parse(game_id)
        tz = datetime.now(UTC).astimezone().tzinfo
        return datetime.fromtimestamp(u.timestamp().int / 1000, tz=tz).strftime("%Y%m%d%H%M%S%f")[:-3]
    except Exception:
        # ULID解析失敗時は現在時刻をフォールバック
        return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]


class AnalysisTracker:
    """トーク分析を行いanalysis.ymlを生成するクラス."""

    def __init__(
        self,
        config: dict[str, Any],
        agent_name: str,
        game_id: str,
        agent_logger=None,
    ) -> None:
        """初期化."""
        self.config = config
        self.agent_name = agent_name
        self.game_id = game_id
        self.packet_idx = 0
        self.agent_logger = agent_logger
        
        # 分析履歴を保存する辞書
        # {packet_idx: [{"content": ..., "type": ..., "from": ..., "to": ..., "credibility": ...}, ...]}
        self.analysis_history: dict[int, list[dict[str, Any]]] = {}
        
        # 既に解析済みのトークを一意キーで保持（重複解析を防ぐ）
        self.seen_talk_keys: set[str] = set()
        
        # 互換のため残すが、重複判定は seen_talk_keys を主とする
        self.last_analyzed_talk_count = 0
        
        # LLMクライアントの初期化
        self.llm_model = None
        self._initialize_llm()
        
        # 出力ディレクトリの設定
        self._setup_output_directory()
        
        # 初期化時点で空ファイルの先行タッチ
        self._ensure_dirs_and_touch_outputs()
    
    def _initialize_llm(self) -> None:
        """LLMクライアントを初期化."""
        from pathlib import Path
        load_dotenv(Path(__file__).parent.joinpath("./../../../../config/.env"))
        
        model_type = str(self.config.get("llm", {}).get("type", "openai"))
        print(f"[AnalysisTracker] Initializing LLM with type: {model_type}")
        
        try:
            match model_type:
                case "openai":
                    api_key = os.environ.get("OPENAI_API_KEY", "")
                    if not api_key:
                        print(f"[AnalysisTracker] Warning: OPENAI_API_KEY not found")
                    self.llm_model = ChatOpenAI(
                        model=str(self.config.get("openai", {}).get("model", "gpt-4o")),
                        temperature=float(self.config.get("openai", {}).get("temperature", 0.7)),
                        api_key=SecretStr(api_key),
                    )
                case "google":
                    api_key = os.environ.get("GOOGLE_API_KEY", "")
                    if not api_key:
                        print(f"[AnalysisTracker] Warning: GOOGLE_API_KEY not found")
                    self.llm_model = ChatGoogleGenerativeAI(
                        model=str(self.config.get("google", {}).get("model", "gemini-2.0-flash-lite")),
                        temperature=float(self.config.get("google", {}).get("temperature", 0.7)),
                        api_key=SecretStr(api_key),
                    )
                case "ollama":
                    self.llm_model = ChatOllama(
                        model=str(self.config.get("ollama", {}).get("model", "llama3.1")),
                        temperature=float(self.config.get("ollama", {}).get("temperature", 0.7)),
                        base_url=str(self.config.get("ollama", {}).get("base_url", "http://localhost:11434")),
                    )
                case _:
                    print(f"[AnalysisTracker] Unknown LLM type: {model_type}")
                    self.llm_model = None
            
            if self.llm_model:
                print(f"[AnalysisTracker] LLM initialized successfully: {type(self.llm_model).__name__}")
            else:
                print(f"[AnalysisTracker] LLM not initialized")
                
        except Exception as e:
            print(f"[AnalysisTracker] Failed to initialize LLM: {e}")
            import traceback
            traceback.print_exc()
            self.llm_model = None
    
    def _setup_output_directory(self) -> None:
        """出力ディレクトリの設定."""
        # プロジェクトルートを解決
        project_root = _resolve_project_root(Path(__file__).resolve())
        
        # game_idをそのまま使用（ゲームごとに1つのディレクトリ）
        # <project_root>/info/bdi_info/micro_bdi/<game_id>/<agent_name>/analysis.yml
        self.output_dir = (
            project_root
            / "info" / "bdi_info" / "micro_bdi"
            / self.game_id / self.agent_name
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_file = self.output_dir / "analysis.yml"
        
        # 可観測性のためのログ出力
        print(f"[AnalysisTracker] analysis_file abs path = {self.analysis_file.resolve()}")
        print(f"[AnalysisTracker] game_id = {self.game_id}, agent_name = {self.agent_name}")
    
    def _talk_key(self, t: Talk) -> str:
        """トークの一意キーを生成して重複を防ぐ."""
        # (day, idx) があればそれをキーに、なければフォールバック
        day = getattr(t, "day", -1)
        idx = getattr(t, "idx", None)
        if idx is not None:
            return f"{day}:{idx}"
        turn = getattr(t, "turn", -1)
        agent = getattr(t, "agent", "")
        import hashlib
        h = hashlib.sha1(str(getattr(t, "text", "")).encode("utf-8", "ignore")).hexdigest()[:16]
        return f"{day}:{turn}:{agent}:{h}"
    
    def _is_meaningful_other_utterance(self, t: Talk) -> bool:
        """有意味な他者の発話かどうかを判定."""
        agent = getattr(t, "agent", "unknown")
        text = getattr(t, "text", "")
        txt = str(text).strip()
        
        if not text or txt == "":
            return False
        
        if txt in {"Skip", "Over"}:
            return False
            
        if getattr(t, "skip", False) or getattr(t, "over", False):
            return False
            
        # 自分の発話は解析対象外
        if agent == self.agent_name:
            return False
        return True
    
    def analyze_talk(
        self,
        talk_history: list[Talk],
        info: Info,
        request_count: int | None = None,  # backward compatibility (ignored)
        **_: Any,                           # ignore unexpected kwargs
    ) -> int:
        """トーク履歴を分析してanalysis.ymlを更新し、今回追加件数を返す."""
        if not talk_history:
            print(f"[AnalysisTracker] analyze_talk: talk_history is empty/None")
            return 0
        
        print(f"[AnalysisTracker] Analyzing talk for {self.agent_name}")
        print(f"[AnalysisTracker] Total talk_history: {len(talk_history)}")
        print(f"[AnalysisTracker] Seen talk keys count: {len(self.seen_talk_keys)}")
        
        # Fixture設定を取得
        cfg = self._get_fixture_config()
        if cfg["enable"]:
            print(f"[AnalysisTracker] Fixture mode ENABLED: output={cfg['output_file']}, max={cfg['max_per_call']}, apply_to={cfg['apply_to_agents']}")
        
        self.packet_idx += 1
        analysis_entries: list[dict[str, Any]] = []
        
        # 全履歴から候補を抽出（seen登録は遅延）
        candidates: list[tuple[str, str, Any, Any]] = []  # (key, kind, talk/proxy, origin_or_None)
        for t in talk_history:
            k = self._talk_key(t)
            if k in self.seen_talk_keys:
                continue
            
            # Fixture条件判定：enable & (対象テキスト or skip/over)
            is_self = (getattr(t, "agent", "") == self.agent_name)
            apply_ok = (cfg["apply_to_agents"] == "all") or (not is_self)
            txt = str(getattr(t, "text", "") or "").strip()
            is_target_text = txt in set(cfg["rewrite_targets"])
            is_flagged = bool(getattr(t, "skip", False) or getattr(t, "over", False))
            
            # 解析可能かどうかを判定
            analyzable = False
            
            if cfg["enable"] and apply_ok and (is_target_text or is_flagged):
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
                # 通常の解析対象
                candidates.append((k, "real", t, None))
                analyzable = True
            
            # 解析不可能なものは即座にseen登録（次回スキップ）
            if not analyzable:
                self.seen_talk_keys.add(k)
        
        if not candidates:
            print(f"[AnalysisTracker] No analyzable talks in this call")
            # 互換のため更新
            self.last_analyzed_talk_count = len(talk_history)
            
            # ハートビート: 空候補でもトレースファイルが空にならないよう保証
            if cfg["enable"] and cfg.get("trace_file"):
                self._record_heartbeat_trace(cfg, "empty_candidates")
            
            return 0
        
        # max_per_call制限を適用
        process_count = min(len(candidates), cfg["max_per_call"] if cfg["enable"] else len(candidates))
        print(f"[AnalysisTracker] Found {len(candidates)} candidates, processing {process_count}")
        
        # 処理する候補のみを解析
        processed_keys = []
        for i, (key, kind, talk, origin) in enumerate(candidates[:process_count]):
            agent_name = getattr(talk, "agent", "unknown")
            text = getattr(talk, "text", "")
            print(f"[AnalysisTracker] Analyzing ({kind}) talk {i+1}/{process_count}: {agent_name} - {text}")
            
            # スタブ先行保存: LLM分析前にpending状態で保存
            self._save_stub_entry_before_llm(talk, kind)
            
            # 発話内容の分析
            analysis_entry = self._analyze_talk_entry(talk, info)
            if analysis_entry:
                analysis_entries.append(analysis_entry)
                print(f"[AnalysisTracker] Analysis entry created ({kind}): {analysis_entry['type']} from {analysis_entry['from']}")
                
                # Fixtureモードの場合、トレースを記録
                if kind == "fixture" and cfg.get("trace_file"):
                    self._record_fixture_trace(origin, talk, cfg)
            else:
                print(f"[AnalysisTracker] No analysis entry created for talk: {text}")
            
            # 処理済みキーをリストに追加
            processed_keys.append(key)
        
        # 処理した分だけseen登録（残りは次回に繰り越し）
        for key in processed_keys:
            self.seen_talk_keys.add(key)
        
        if len(candidates) > process_count:
            print(f"[AnalysisTracker] {len(candidates) - process_count} candidates deferred to next call")
        
        # 分析結果を履歴に保存
        if analysis_entries:
            self.analysis_history[self.packet_idx] = analysis_entries
            print(f"[AnalysisTracker] Saved {len(analysis_entries)} analysis entries for packet_idx {self.packet_idx}")
        else:
            print(f"[AnalysisTracker] No analysis entries to save")
        
        # 追跡値を更新
        self.last_analyzed_talk_count = len(talk_history)
        print(f"[AnalysisTracker] newly added entries in this call: {len(analysis_entries)}")
        
        # --- タイムアウト対策: 分析直後に即座保存 ---
        try:
            print(f"[AnalysisTracker] === save_analysis 開始 (immediate) ===")
            self.save_analysis()
            if cfg["enable"] and cfg.get("trace_file"):
                self.save_fixture_trace()
            print(f"[AnalysisTracker] === save_analysis 終了 (immediate) ===")
        except Exception as e:
            print(f"[AnalysisTracker] Immediate save failed: {e}")
            if self.agent_logger:
                self.agent_logger.logger.exception(f"Immediate save failed: {e}")
        
        return len(analysis_entries)
    
    def _analyze_talk_entry(
        self,
        talk: Talk,
        info: Info,
    ) -> dict[str, Any] | None:
        """個別のトーク発話を分析."""
        if not talk.text or talk.text.strip() == "":
            print(f"[AnalysisTracker] Skipping empty talk from {talk.agent}")
            return None
        
        # "Over"発話は分析対象に含める
        if talk.text.strip() == "Over":
            print(f"[AnalysisTracker] Processing 'Over' talk from {talk.agent}")
            return {
                "content": talk.text,
                "type": "null",
                "from": talk.agent,
                "to": "null",
                "credibility": 0.0,
            }
        
        # 基本情報を抽出
        content = talk.text
        from_agent = talk.agent
        
        print(f"[AnalysisTracker] Processing talk from {from_agent}: '{content}'")
        print(f"[AnalysisTracker] LLM model available: {self.llm_model is not None}")
        
        # LLMを使って発話を分析
        message_type = self._analyze_message_type(content, info)
        to_agents = self._analyze_target_agents(content, info)
        credibility = self._analyze_credibility(content, info, from_agent)
        
        print(f"[AnalysisTracker] Analysis results - type: {message_type}, to: {to_agents}, credibility: {credibility}")
        
        return {
            "content": content,
            "type": message_type,
            "from": from_agent,
            "to": to_agents,
            "credibility": credibility,
        }
    
    def _analyze_message_type(
        self,
        content: str,
        info: Info,
    ) -> str:
        """発話のタイプを分析."""
        # LLMを使って発話タイプを分析
        if not self.llm_model:
            if self.agent_logger:
                self.agent_logger.llm_error("message_type_analysis", "LLM model not available")
            return "null"
        
        try:
            from jinja2 import Template
            from langchain_core.messages import HumanMessage
            from langchain_core.output_parsers import StrOutputParser
            
            # プロンプトテンプレートの作成（config → デフォルト）
            prompt_template = self.config.get("prompt", {}).get("analyze_message_type", "") if self.config else ""
            if not prompt_template:
                prompt_template = DEFAULT_PROMPTS.analyze_message_type
                if self.agent_logger:
                    self.agent_logger.logger.info("Using default prompt for analyze_message_type")
            
            template = Template(prompt_template)
            prompt = template.render(
                content=content,
                agent_names=list(info.status_map.keys()),
            )
            
            # LLMに問い合わせ
            messages = [HumanMessage(content=prompt)]
            response = (self.llm_model | StrOutputParser()).invoke(messages)
            
            # LLMやり取りをログ出力
            if self.agent_logger:
                model_info = f"{type(self.llm_model).__name__}"
                self.agent_logger.llm_interaction("message_type_analysis", prompt, response, model_info)
            
            # 応答からタイプを抽出
            response = response.strip().lower()
            
            # 優先順位順でタイプを判定
            if "co" in response:
                result = "co"
            elif "question" in response:
                result = "question"
            elif "negative" in response:
                result = "negative"
            elif "positive" in response:
                result = "positive"
            else:
                result = "null"
            
            if self.agent_logger:
                self.agent_logger.logger.info(f"Message type analysis result: '{content}' -> '{result}'")
            
            return result
                
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.llm_error("message_type_analysis", str(e), prompt if 'prompt' in locals() else None)
            return "null"
    
    def _analyze_target_agents(
        self,
        content: str,
        info: Info,
    ) -> str:
        """発話の対象エージェントを分析."""
        # LLMを使って発話対象を分析
        if not self.llm_model:
            if self.agent_logger:
                self.agent_logger.llm_error("target_agents_analysis", "LLM model not available")
            return "null"
        
        try:
            from jinja2 import Template
            from langchain_core.messages import HumanMessage
            from langchain_core.output_parsers import StrOutputParser
            
            # プロンプトテンプレートの作成（config → デフォルト）
            prompt_template = self.config.get("prompt", {}).get("analyze_target_agents", "") if self.config else ""
            if not prompt_template:
                prompt_template = DEFAULT_PROMPTS.analyze_target_agents
                if self.agent_logger:
                    self.agent_logger.logger.info("Using default prompt for analyze_target_agents")
            
            template = Template(prompt_template)
            prompt = template.render(
                content=content,
                agent_names=list(info.status_map.keys()),
            )
            
            # LLMに問い合わせ
            messages = [HumanMessage(content=prompt)]
            response = (self.llm_model | StrOutputParser()).invoke(messages)
            
            # LLMやり取りをログ出力
            if self.agent_logger:
                model_info = f"{type(self.llm_model).__name__}"
                self.agent_logger.llm_interaction("target_agents_analysis", prompt, response, model_info)
            
            # 応答から対象エージェントを抽出
            response = response.strip()
            
            # 特定のエージェント名が含まれているかチェック
            mentioned_agents = []
            for agent_name in info.status_map.keys():
                if agent_name in response:
                    mentioned_agents.append(agent_name)
            
            if mentioned_agents:
                result = ",".join(mentioned_agents)
            elif "all" in response.lower() or "全体" in response:
                result = "all"
            else:
                result = "null"
            
            if self.agent_logger:
                self.agent_logger.logger.info(f"Target agents analysis result: '{content}' -> '{result}'")
            
            return result
                
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.llm_error("target_agents_analysis", str(e), prompt if 'prompt' in locals() else None)
            return "null"
    
    def _analyze_credibility(
        self,
        content: str,
        info: Info,
        from_agent: str,
    ) -> float:
        """発話の信憑性を分析し、加重平均を計算."""
        # LLMを使って信憑性スコアを取得
        raw_scores = self._get_credibility_scores(content, info)
        if not raw_scores:
            return 0.0
        
        # macro_belief.ymlから重みを取得
        weights = self._get_statement_bias_weights(from_agent)
        if not weights:
            # 重みが取得できない場合は単純平均
            return sum(raw_scores.values()) / len(raw_scores)
        
        # 加重平均を計算
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, score in raw_scores.items():
            weight = weights.get(key, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0.0:
            return 0.0
        
        return round(weighted_sum / total_weight, 2)
    
    def _get_credibility_scores(
        self,
        content: str,
        info: Info,
    ) -> dict[str, float]:
        """LLMを使って信憑性の4つのスコアを取得."""
        if not self.llm_model:
            if self.agent_logger:
                self.agent_logger.llm_error("credibility_analysis", "LLM model not available")
            return {}
        
        try:
            from jinja2 import Template
            from langchain_core.messages import HumanMessage
            from langchain_core.output_parsers import StrOutputParser
            
            # プロンプトテンプレートの作成（config → デフォルト）
            prompt_template = self.config.get("prompt", {}).get("analyze_credibility", "") if self.config else ""
            if not prompt_template:
                prompt_template = DEFAULT_PROMPTS.analyze_credibility
                if self.agent_logger:
                    self.agent_logger.logger.info("Using default prompt for analyze_credibility")
            
            template = Template(prompt_template)
            prompt = template.render(
                content=content,
                agent_names=list(info.status_map.keys()),
            )
            
            # LLMに問い合わせ
            messages = [HumanMessage(content=prompt)]
            response = (self.llm_model | StrOutputParser()).invoke(messages)
            
            # LLMやり取りをログ出力
            if self.agent_logger:
                model_info = f"{type(self.llm_model).__name__}"
                self.agent_logger.llm_interaction("credibility_analysis", prompt, response, model_info)
            
            # 応答からスコアを抽出
            scores = {}
            for line in response.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    try:
                        scores[key] = float(value.strip())
                    except ValueError:
                        continue
            
            # 期待される4つのキーが全て存在するかチェック
            expected_keys = [
                "logical_consistency",
                "specificity_and_detail", 
                "intuitive_depth",
                "clarity_and_conciseness"
            ]
            
            if all(key in scores for key in expected_keys):
                if self.agent_logger:
                    self.agent_logger.logger.info(f"Credibility analysis scores: '{content}' -> {scores}")
                return scores
            else:
                if self.agent_logger:
                    self.agent_logger.logger.warning(f"Incomplete credibility scores for '{content}': {scores}")
                return {}
                
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.llm_error("credibility_analysis", str(e), prompt if 'prompt' in locals() else None)
            return {}
    
    def _get_statement_bias_weights(self, agent_name: str) -> dict[str, float]:
        """macro_belief.ymlからstatement_biasの重みを取得."""
        try:
            # macro_belief.ymlのパスを構築
            macro_belief_path = (
                Path("info") / "bdi_info" / "macro_bdi" / self.game_id / agent_name / "macro_belief.yml"
            )
            
            if not macro_belief_path.exists():
                return {}
            
            # YAMLファイルを読み込み
            with open(macro_belief_path, 'r', encoding='utf-8') as f:
                macro_belief_data = yaml.safe_load(f)
            
            # statement_biasの重みを取得
            statement_bias = (
                macro_belief_data
                .get("macro_belief", {})
                .get("cognitive_bias", {})
                .get("statement_bias", {})
            )
            
            return statement_bias
            
        except Exception:
            return {}
    
    def save_analysis(self) -> None:
        """analysis.ymlファイルに保存."""
        # Fixture設定を取得
        cfg = self._get_fixture_config()
        
        # 保存先の決定（Fixture有効時は分離ファイルへ）
        target_path = self.output_dir / (cfg["output_file"] if cfg["enable"] else "analysis.yml")
        
        # デバッグログ：保存前の状態を出力
        print(f"[AnalysisTracker] === save_analysis 開始 ===")
        print(f"[AnalysisTracker] agent_name: {self.agent_name}")
        print(f"[AnalysisTracker] game_id: {self.game_id}")
        print(f"[AnalysisTracker] Fixture mode: {'ENABLED' if cfg['enable'] else 'DISABLED'}")
        print(f"[AnalysisTracker] Target file: {target_path.name}")
        print(f"[AnalysisTracker] analysis_history keys: {list(self.analysis_history.keys())}")
        
        # 連番マップ化（1始まり）
        all_entries = []
        # キーを文字列に変換して安全にソート
        sorted_keys = sorted(self.analysis_history.keys(), key=lambda x: str(x))
        for packet_idx in sorted_keys:
            entries = self.analysis_history[packet_idx] or []
            print(f"[AnalysisTracker]   packet_idx {packet_idx}: {len(entries)} entries")
            all_entries.extend(entries)
        
        print(f"[AnalysisTracker] 合計エントリ数: {len(all_entries)}")
        
        if all_entries:
            # 1始まりの連番キーでデータを構築
            data = {
                i + 1: {
                    "content": e.get("content", ""),
                    "type": e.get("type", "null"),
                    "from": e.get("from", "unknown"),
                    "to": e.get("to", "null"),
                    "credibility": float(e.get("credibility", 0.0)),
                }
                for i, e in enumerate(all_entries)
            }
        else:
            # 空の場合でも空辞書を出力（0バイトファイルは作らない）
            data = {}
        
        # 原子的保存: 一時ファイルに書き込んでから置換
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                delete=False,
                dir=str(self.output_dir),
                prefix="analysis_",
                suffix=".tmp"
            ) as tmp:
                # yaml.safe_dumpで安全に出力
                yaml.safe_dump(
                    data,
                    tmp,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False
                )
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            
            # 原子的にファイルを置換（Windows/Linux両対応）
            os.replace(str(tmp_path), str(target_path))
            
        except Exception as e:
            print(f"[AnalysisTracker] ERROR during file save: {e}")
            import traceback
            traceback.print_exc()
            if self.agent_logger:
                self.agent_logger.logger.error(f"Failed to save analysis file: {e}")
            raise
        
        # 保存成功のログ出力（可観測性）
        file_size = target_path.stat().st_size
        print(f"[AnalysisTracker] saved: {target_path.resolve()} size={file_size}")
        print(f"[AnalysisTracker] 保存したデータキー数: {len(data) if 'data' in locals() else 0}")
        
        # 直後に読み戻して確認（拡張版：構造も検証）
        try:
            # バイナリ読み戻し
            with open(target_path, "rb") as rf:
                blob = rf.read()
            print(f"[AnalysisTracker] readback right after save: {len(blob)} bytes, head={blob[:120]!r}")
            
            # YAML構造検証
            self._verify_saved_file(target_path, len(data) if 'data' in locals() else 0)
            
        except Exception as e:
            print(f"[AnalysisTracker] readback failed: {e}")
        
        # 下流処理の実行フラグを取得
        ds_flags = self._get_downstream_flags()
        
        # Fixture有効時は既定で下流処理をSKIP（環境変数で上書き可能）
        if cfg["enable"]:
            # Fixture有効時のデフォルトはFalse、環境変数があればそれを優先
            if "ANALYSIS_UPDATE_SELECT_SENTENCE" not in os.environ:
                ds_flags["update_select_sentence"] = False
            if "ANALYSIS_UPDATE_INTENTION" not in os.environ:
                ds_flags["update_intention"] = False
        
        before = target_path.stat().st_size
        
        # select_sentence更新
        if ds_flags.get("update_select_sentence", True):
            print(f"[AnalysisTracker] Downstream(select_sentence): RUN")
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
        
        # intention更新
        if ds_flags.get("update_intention", True):
            print(f"[AnalysisTracker] Downstream(intention): RUN")
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
        
        print(f"[AnalysisTracker] === save_analysis 終了 ===")
    
    def save_fixture_trace(self) -> None:
        """Fixtureトレースファイル全体を保存（冪等性保証のため）."""
        cfg = self._get_fixture_config()
        if not cfg["enable"] or not cfg.get("trace_file"):
            return
        
        trace_path = self.output_dir / cfg["trace_file"]
        
        print(f"[AnalysisTracker] === save_fixture_trace 開始 ===")
        
        # 既存のトレースデータを読み込み（あれば）
        existing_traces = {}
        if trace_path.exists():
            try:
                with open(trace_path, 'r', encoding='utf-8') as f:
                    existing_traces = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[AnalysisTracker] Failed to read existing trace: {e}")
        
        # アトミック保存（空でも必ず書く）
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                delete=False,
                dir=str(self.output_dir),
                prefix="trace_",
                suffix=".tmp"
            ) as tmp:
                if not existing_traces:
                    tmp.write("{}\n")
                else:
                    yaml.safe_dump(
                        existing_traces,
                        tmp,
                        allow_unicode=True,
                        sort_keys=False,
                        default_flow_style=False
                    )
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            
            # 原子的にファイルを置換
            os.replace(str(tmp_path), str(trace_path))
            
        except Exception as e:
            print(f"[AnalysisTracker] ERROR during trace save: {e}")
            if self.agent_logger:
                self.agent_logger.logger.error(f"Failed to save trace file: {e}")
            raise
        
        print(f"[AnalysisTracker] saved trace: {trace_path.resolve()}")
        
        # トレースファイルの読み戻し検証
        try:
            trace_size = trace_path.stat().st_size
            print(f"[AnalysisTracker] trace file size: {trace_size} bytes")
            
            # YAML構造検証
            self._verify_saved_file(trace_path, len(existing_traces))
            
        except Exception as e:
            print(f"[AnalysisTracker] trace file verification failed: {e}")
        
        print(f"[AnalysisTracker] === save_fixture_trace 終了 ===")
    
    def force_save_with_status(self) -> None:
        """強制的に現在の状態をファイルに保存（デバッグ用）."""
        print(f"[AnalysisTracker] Force saving analysis status:")
        print(f"  - agent_name: {self.agent_name}")
        print(f"  - game_id: {self.game_id}")
        print(f"  - packet_idx: {self.packet_idx}")
        print(f"  - last_analyzed_talk_count: {self.last_analyzed_talk_count}")
        print(f"  - LLM model available: {self.llm_model is not None}")
        print(f"  - analysis_history keys: {list(self.analysis_history.keys())}")
        
        # 通常の保存処理を実行
        self.save_analysis()
    
    def _update_select_sentence(self) -> None:
        """select_sentence.ymlを更新."""
        try:
            from .select_sentence import SelectSentenceTracker
            
            # SelectSentenceTrackerを使って更新
            tracker = SelectSentenceTracker(self.agent_name, self.game_id)
            tracker.process_select_sentence()
        except ImportError:
            # select_sentence モジュールが存在しない場合は無視
            pass
        except Exception as e:
            print(f"Failed to update select_sentence: {e}")
    
    def _update_intention(self) -> None:
        """intention.ymlを更新."""
        try:
            from .intention_tracker import IntentionTracker
            
            # IntentionTrackerを使って更新
            tracker = IntentionTracker(self.config, self.agent_name, self.game_id)
            tracker.process_intention()
        except ImportError:
            # intention_tracker モジュールが存在しない場合は無視
            pass
        except Exception as e:
            print(f"Failed to update intention: {e}")
    
    # -------- Fixture mode helpers --------
    def _env_bool(self, name: str, default: bool = False) -> bool:
        """環境変数を真偽値として解釈."""
        val = os.environ.get(name, "").lower()
        if val in {"1", "true", "yes", "on"}:
            return True
        elif val in {"0", "false", "no", "off"}:
            return False
        return default
    
    def _env_list(self, name: str, sep: str = ",", default: list | None = None) -> list[str]:
        """環境変数をリストとして解釈."""
        val = os.environ.get(name, "")
        if not val:
            return default or []
        return [s.strip() for s in val.split(sep) if s.strip()]
    
    def _get_fixture_config(self) -> dict[str, Any]:
        """Fixture設定を取得（config.yml + 環境変数）."""
        # config.ymlからベース設定を読み込み（存在しない場合は空）
        cfg_base = self.config.get("analysis", {}).get("fixture_mode", {}) if self.config else {}
        
        # デフォルト値を設定（config.ymlが無くても動作）
        cfg = {
            "enable": cfg_base.get("enable", False),
            "output_file": cfg_base.get("output_file", "analysis_test.yml"),
            "trace_file": cfg_base.get("trace_file", "analysis_fixture_trace.yml"),
            "rewrite_targets": cfg_base.get("rewrite_targets", ["Skip", "Over"]),
            "max_per_call": cfg_base.get("max_per_call", 999),  # Default to high value for env-only usage
            "apply_to_agents": cfg_base.get("apply_to_agents", "others"),
            "utterances": {
                "default": cfg_base.get("utterances", {}).get("default", DEFAULT_FIXTURE_UTTERANCES[:]),
                "by_agent": cfg_base.get("utterances", {}).get("by_agent", {})
            }
        }
        
        # 環境変数で上書き（優先度最高）
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
        """Fixture用の置換テキストを選択."""
        agent = getattr(talk, "agent", "")
        
        # エージェント別の候補を優先
        candidates = cfg["utterances"].get("by_agent", {}).get(agent, [])
        if not candidates:
            candidates = cfg["utterances"].get("default", [])
        
        if not candidates:
            return None
        
        # 安定したローテーション（day, turn, idxを元に選択）
        day = getattr(talk, "day", 0)
        turn = getattr(talk, "turn", 0)
        idx = getattr(talk, "idx", 0) or 0
        
        # ハッシュ値を計算して候補から選択
        hash_val = (day * 1000 + turn * 10 + idx) % len(candidates)
        return candidates[hash_val]
    
    def _record_fixture_trace(self, origin: Any, proxy: Any, cfg: dict[str, Any]) -> None:
        """Fixture置換の痕跡を記録."""
        trace_file = cfg.get("trace_file")
        if not trace_file:
            return
        
        trace_path = self.output_dir / trace_file
        
        # トレースエントリを作成
        day = getattr(origin, "day", -1)
        idx = getattr(origin, "idx", None)
        turn = getattr(origin, "turn", -1)
        agent = getattr(origin, "agent", "unknown")
        original_text = getattr(origin, "text", "")
        replaced_text = getattr(proxy, "text", "")
        
        # キーを生成
        import hashlib
        text_hash = hashlib.sha1(str(original_text).encode("utf-8", "ignore")).hexdigest()[:8]
        if idx is not None:
            key = f"{day}:{idx}:{text_hash}"
        else:
            key = f"{day}:{turn}:{text_hash}"
        
        # トレースデータ
        trace_entry = {
            key: {
                "from_agent": agent,
                "original": original_text,
                "replaced": replaced_text,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # 既存のトレースデータを読み込み（あれば）
        existing_traces = {}
        if trace_path.exists():
            try:
                with open(trace_path, 'r', encoding='utf-8') as f:
                    existing_traces = yaml.safe_load(f) or {}
            except Exception:
                pass
        
        # マージして保存
        existing_traces.update(trace_entry)
        
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                delete=False,
                dir=str(self.output_dir),
                prefix="trace_",
                suffix=".tmp"
            ) as tmp:
                yaml.safe_dump(
                    existing_traces,
                    tmp,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False
                )
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            
            os.replace(str(tmp_path), str(trace_path))
            print(f"[AnalysisTracker] Fixture trace recorded: {key}")
        except Exception as e:
            print(f"[AnalysisTracker] Failed to record fixture trace: {e}")
    
    def _get_downstream_flags(self) -> dict[str, bool]:
        """下流処理の実行フラグを取得."""
        # デフォルトは True（実行）
        flags = {
            "update_select_sentence": True,
            "update_intention": True
        }
        
        # 環境変数で上書き
        if "ANALYSIS_UPDATE_SELECT_SENTENCE" in os.environ:
            flags["update_select_sentence"] = self._env_bool("ANALYSIS_UPDATE_SELECT_SENTENCE", True)
        if "ANALYSIS_UPDATE_INTENTION" in os.environ:
            flags["update_intention"] = self._env_bool("ANALYSIS_UPDATE_INTENTION", True)
        
        return flags
    
    def _ensure_dirs_and_touch_outputs(self) -> None:
        """初期化時点で出力先を先行作成し、空ファイルをタッチしておく."""
        # 出力ディレクトリは既に _setup_output_directory() で作成済み
        
        # Fixture設定を取得
        cfg = self._get_fixture_config()
        
        # 作成すべきファイルパスを決定
        paths = []
        trace_path = None
        
        if cfg["enable"]:
            # Fixture有効時は analysis_test.yml と trace ファイル
            paths.append(self.output_dir / cfg["output_file"])
            if cfg.get("trace_file"):
                trace_path = self.output_dir / cfg["trace_file"]
                paths.append(trace_path)
        else:
            # 通常時は analysis.yml
            paths.append(self.analysis_file)
        
        # 空ファイルを先行作成
        for path in paths:
            if not path.exists():
                try:
                    if path == trace_path:
                        # トレースファイルの場合：ハートビートエントリで非空保証
                        self._create_trace_with_heartbeat(path)
                    else:
                        # 通常ファイルの場合：空の辞書
                        with open(path, "w", encoding="utf-8") as f:
                            f.write("{}\n")
                    print(f"[AnalysisTracker] Pre-touched: {path}")
                except Exception as e:
                    print(f"[AnalysisTracker] Failed to pre-touch {path}: {e}")
    
    def _create_trace_with_heartbeat(self, trace_path) -> None:
        """トレースファイルをハートビートエントリで初期化して非空保証."""
        # ハートビートエントリを作成
        meta_key = f"meta:tracker_init:{int(time.time())}:{random.randint(1000, 9999)}"
        heartbeat_entry = {
            meta_key: {
                "from_agent": None,
                "original": "",
                "replaced": "",
                "timestamp": datetime.now().isoformat(),
                "reason": "tracker_init"
            }
        }
        
        try:
            with open(trace_path, "w", encoding="utf-8") as f:
                yaml.dump(heartbeat_entry, f, default_flow_style=False, allow_unicode=True)
            print(f"[AnalysisTracker] Created trace with heartbeat: {trace_path}")
        except Exception as e:
            print(f"[AnalysisTracker] Failed to create heartbeat trace {trace_path}: {e}")
            # フォールバック: 空の辞書で作成
            with open(trace_path, "w", encoding="utf-8") as f:
                f.write("{}\n")
    
    def _record_heartbeat_trace(self, cfg: dict[str, Any], reason: str) -> None:
        """ハートビートトレースを記録して空ファイル化を防ぐ."""
        trace_file = cfg.get("trace_file")
        if not trace_file:
            return
        
        trace_path = self.output_dir / trace_file
        
        # ハートビートエントリを作成
        meta_key = f"meta:heartbeat:{reason}:{int(time.time())}:{random.randint(1000, 9999)}"
        heartbeat_entry = {
            "from_agent": None,
            "original": "",
            "replaced": "",
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        }
        
        # 既存のトレースデータを読み込み（あれば）
        existing_traces = {}
        if trace_path.exists():
            try:
                with open(trace_path, 'r', encoding='utf-8') as f:
                    existing_traces = yaml.safe_load(f) or {}
            except Exception:
                pass
        
        # ハートビートエントリをマージ
        existing_traces[meta_key] = heartbeat_entry
        
        # 原子的書き込み
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8", 
                delete=False,
                dir=str(self.output_dir),
                prefix="heartbeat_",
                suffix=".tmp"
            ) as tmp_file:
                yaml.dump(existing_traces, tmp_file, default_flow_style=False, allow_unicode=True)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                
            os.replace(tmp_file.name, trace_path)
            print(f"[AnalysisTracker] Recorded heartbeat trace: {reason}")
            
        except Exception as e:
            print(f"[AnalysisTracker] Failed to record heartbeat trace: {e}")
            # クリーンアップ
            try:
                if 'tmp_file' in locals() and os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
            except Exception:
                pass
    
    def _save_stub_entry_before_llm(self, talk: Talk, kind: str) -> None:
        """LLM分析前にスタブエントリを保存してタイムアウト対策."""
        # Fixture設定を取得
        cfg = self._get_fixture_config()
        
        # スタブエントリを作成
        agent_name = getattr(talk, "agent", "unknown")
        text = getattr(talk, "text", "")
        
        # packet_idxを生成
        day = getattr(talk, "day", -1)
        idx = getattr(talk, "idx", None)
        turn = getattr(talk, "turn", -1)
        if idx is not None:
            packet_idx = f"{day}_{idx}"
        else:
            packet_idx = f"{day}_{turn}_{hashlib.sha1(text.encode('utf-8', 'ignore')).hexdigest()[:8]}"
        
        stub_entry = {
            "content": f"[PENDING] Analyzing: {text[:50]}...",
            "type": "pending_analysis", 
            "from": agent_name,
            "to": "all",
            "credibility": 0.0,
            "timestamp": datetime.now().isoformat(),
            "status": "pending_llm_analysis",
            "original_text": text,
            "analysis_kind": kind
        }
        
        # 一時的にanalysis_historyに保存
        if packet_idx not in self.analysis_history:
            self.analysis_history[packet_idx] = []
        self.analysis_history[packet_idx].append(stub_entry)
        
        # 即座にファイル保存（タイムアウト対策）
        try:
            self.save_analysis()
            if cfg["enable"] and cfg.get("trace_file"):
                # スタブでもトレースに記録
                self._record_heartbeat_trace(cfg, f"stub_saved_{kind}")
            print(f"[AnalysisTracker] Saved stub entry for {agent_name}: {text[:30]}...")
        except Exception as e:
            print(f"[AnalysisTracker] Failed to save stub entry: {e}")
    
    def _verify_saved_file(self, file_path, expected_entry_count: int) -> None:
        """保存されたファイルの構造検証."""
        try:
            # ファイルサイズチェック
            if not file_path.exists():
                raise FileNotFoundError(f"Saved file does not exist: {file_path}")
            
            file_size = file_path.stat().st_size
            if file_size == 0:
                raise ValueError(f"Saved file is empty: {file_path}")
            
            # YAML読み込みテスト
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_data = yaml.safe_load(f)
            
            # データ構造チェック
            if loaded_data is None:
                loaded_data = {}
            
            if not isinstance(loaded_data, dict):
                raise ValueError(f"Saved file does not contain valid dict structure: {type(loaded_data)}")
            
            # エントリ数チェック
            actual_count = len(loaded_data)
            if actual_count != expected_entry_count:
                print(f"[AnalysisTracker] WARNING: Entry count mismatch. Expected: {expected_entry_count}, Actual: {actual_count}")
            
            print(f"[AnalysisTracker] File verification passed: {file_path.name} ({file_size} bytes, {actual_count} entries)")
            
        except Exception as e:
            print(f"[AnalysisTracker] File verification FAILED: {file_path} - {e}")
            raise
    
    def sync_with_trace(self) -> None:
        """トレースファイルとの同期を行い、欠損エントリを補完する."""
        cfg = self._get_fixture_config()
        
        if not cfg["enable"] or not cfg.get("trace_file"):
            print(f"[AnalysisTracker] sync_with_trace: Fixture mode disabled, skipping")
            return
        
        trace_path = self.output_dir / cfg["trace_file"]
        analysis_path = self.output_dir / cfg["output_file"]
        
        print(f"[AnalysisTracker] === sync_with_trace 開始 ===")
        print(f"[AnalysisTracker] Trace file: {trace_path}")
        print(f"[AnalysisTracker] Analysis file: {analysis_path}")
        
        # トレースファイルから既存エントリを読み込み
        existing_traces = {}
        if trace_path.exists():
            try:
                with open(trace_path, 'r', encoding='utf-8') as f:
                    existing_traces = yaml.safe_load(f) or {}
                print(f"[AnalysisTracker] Loaded {len(existing_traces)} trace entries")
            except Exception as e:
                print(f"[AnalysisTracker] Failed to load trace file: {e}")
                return
        
        # 分析ファイルから既存エントリを読み込み
        existing_analysis = {}
        if analysis_path.exists():
            try:
                with open(analysis_path, 'r', encoding='utf-8') as f:
                    existing_analysis = yaml.safe_load(f) or {}
                print(f"[AnalysisTracker] Loaded {len(existing_analysis)} analysis entries")
            except Exception as e:
                print(f"[AnalysisTracker] Failed to load analysis file: {e}")
        
        # 現在のanalysis_historyとファイル内容を同期
        sync_needed = False
        
        # analysis_historyに存在してファイルに存在しない項目を検出
        all_current_entries = []
        # キーを文字列に変換して安全にソート
        sorted_keys = sorted(self.analysis_history.keys(), key=lambda x: str(x))
        for packet_idx in sorted_keys:
            entries = self.analysis_history[packet_idx] or []
            all_current_entries.extend(entries)
        
        current_count = len(all_current_entries)
        file_count = len(existing_analysis)
        
        if current_count != file_count:
            print(f"[AnalysisTracker] Count mismatch detected: memory={current_count}, file={file_count}")
            sync_needed = True
        
        # ハートビートエントリ追加（同期の証跡）
        if sync_needed or len(existing_traces) == 0:
            self._record_heartbeat_trace(cfg, "sync_executed")
            print(f"[AnalysisTracker] Added sync heartbeat to trace")
        
        # 必要に応じてファイル再保存
        if sync_needed:
            print(f"[AnalysisTracker] Executing file sync...")
            try:
                self.save_analysis()
                self.save_fixture_trace()
                print(f"[AnalysisTracker] File sync completed")
            except Exception as e:
                print(f"[AnalysisTracker] File sync failed: {e}")
        else:
            print(f"[AnalysisTracker] Files already in sync")
        
        print(f"[AnalysisTracker] === sync_with_trace 終了 ===")
