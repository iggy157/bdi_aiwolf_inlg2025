# agent.py
"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import os
import random
import re
import yaml
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# ★ 追加: 安全化 I/O レイヤ
from llm_safe_agent_io import LLMSafeIO

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread
from utils.bdi.micro_bdi.extract_pairs import extract_pairs_for_agent
from utils.bdi.micro_bdi.affinity_trust_updater import update_affinity_trust_for_agent
from utils.bdi.micro_bdi.talk_history_init import init_talk_history_for_agent, determine_talk_dir
from utils.bdi.micro_bdi.credibility_adjuster import apply_affinity_to_analysis
from utils.bdi.micro_bdi.micro_desire_generator import generate_micro_desire_for_agent
from utils.bdi.micro_bdi.micro_intention_generator import generate_micro_intention_for_agent
from utils.bdi.macro_bdi.macro_belief import generate_macro_belief
from utils.bdi.macro_bdi.macro_desire import generate_macro_desire
# from utils.bdi.macro_bdi.macro_plan import generate_macro_plan  # [macro_plan disabled]
from utils.bdi.micro_bdi.analysis_tracker import AnalysisTracker
from utils.bdi.micro_bdi.micro_belief import write_micro_belief  # ← 追加
from utils.bdi.micro_bdi.self_talk_logger import write_self_talk_for_agent  # ← 追加

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")


class Agent:
    """Base class for agents.

    エージェントの基底クラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """Initialize the agent.

        エージェントの初期化を行う.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role / 役職
        """
        self.config = config
        self.agent_name = name
        self.agent_logger = AgentLogger(config, name, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role

        self.sent_talk_count: int = 0
        self.sent_whisper_count: int = 0
        self.llm_model: BaseChatModel | None = None
        self.llm_message_history: list[BaseMessage] = []

        # AnalysisTrackerの初期化
        self.analysis_tracker: AnalysisTracker | None = None

        load_dotenv(Path(__file__).parent.joinpath("./../../config/.env"))

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to set action timeout.

        アクションタイムアウトを設定するデコレータ.

        Args:
            func (Callable[P, T]): Function to be decorated / デコレート対象の関数

        Returns:
            Callable[P, T]: Function with timeout functionality / タイムアウト機能を追加した関数
        """

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    # =========================
    # LLM 呼び出し: 既存ユーティリティ
    # =========================

    def set_packet(self, packet: Packet) -> None:
        """Set packet information.

        パケット情報をセットする.

        Args:
            packet (Packet): Received packet / 受信したパケット
        """
        self.request = packet.request
        if packet.info:
            self.info = packet.info
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)
        if self.request == Request.INITIALIZE:
            self.talk_history: list[Talk] = []
            self.whisper_history: list[Talk] = []
            self.llm_message_history: list[BaseMessage] = []
        self.agent_logger.logger.debug(packet)

    def get_alive_agents(self) -> list[str]:
        """Get the list of alive agents.

        生存しているエージェントのリストを取得する.

        Returns:
            list[str]: List of alive agent names / 生存エージェント名のリスト
        """
        if not self.info or not getattr(self.info, "status_map", None):
            return []
        alive = []
        for k, v in self.info.status_map.items():
            name_attr = getattr(v, "name", None)
            if (isinstance(v, str) and v == "ALIVE") or (isinstance(name_attr, str) and name_attr == "ALIVE"):
                alive.append(k)
        return alive

    # ---- helper: Request → prompt key (safe) ----
    @staticmethod
    def _request_to_prompt_key(request: Request | None) -> str | None:
        if request is None:
            return None
        # Enum対応
        name = getattr(request, "name", None)
        if isinstance(name, str):
            return name.lower()
        # 文字列風にも対応（保険）
        try:
            s = str(request)
            return s.lower()
        except Exception:
            return None

    def _send_message_to_llm(self, request: Request | None) -> str | None:
        """Send message to LLM and get response.

        LLMにメッセージを送信して応答を取得する.
        """
        key_name = self._request_to_prompt_key(request)
        if key_name is None:
            return None
        if "prompt" not in self.config or key_name not in self.config["prompt"]:
            return None
        prompt_template = self.config["prompt"][key_name]
        if float(self.config["llm"]["sleep_time"]) > 0:
            sleep(float(self.config["llm"]["sleep_time"]))
        key = {
            "info": self.info,
            "setting": self.setting,
            "talk_history": self.talk_history,
            "whisper_history": self.whisper_history,
            "role": self.role,
            "sent_talk_count": self.sent_talk_count,
            "sent_whisper_count": self.sent_whisper_count,
        }
        template: Template = Template(prompt_template)
        prompt = template.render(**key).strip()
        if self.llm_model is None:
            self.agent_logger.logger.error("LLM is not initialized")
            return None
        try:
            self.llm_message_history.append(HumanMessage(content=prompt))
            response = (self.llm_model | StrOutputParser()).invoke(self.llm_message_history)
            self.llm_message_history.append(AIMessage(content=response))

            # Comprehensive LLM interaction logging
            model_info = f"{type(self.llm_model).__name__}"
            self.agent_logger.llm_interaction("agent_main", prompt, response, model_info)

        except Exception as e:
            self.agent_logger.llm_error("agent_main", str(e), prompt if 'prompt' in locals() else None)
            self.agent_logger.logger.exception("Failed to send message to LLM")
            return None
        else:
            return response

    def send_message_to_llm(
        self,
        prompt_key: str,
        extra_vars: dict[str, Any] | None = None,
        *,
        log_tag: str | None = None,
        use_shared_history: bool = False,
    ) -> str | None:
        """Send message to LLM using config-based prompt and return response.

        config['prompt'][prompt_key]をJinja2でレンダリングし、LLMを実行して文字列レスポンスを返す.
        - プロンプトはconfig/config.ymlからのみ取得
        - APIキーはconfig/.envに依存（agent初期化の既存load_dotenvを利用）
        - 既定ではチャット履歴を共有しない（aux系生成物は会話履歴に汚染を与えない）
        """
        if self.llm_model is None:
            self.agent_logger.logger.error("LLM is not initialized")
            return None

        if "prompt" not in self.config or prompt_key not in self.config["prompt"]:
            self.agent_logger.logger.error(f"Prompt key not found: {prompt_key}")
            return None

        prompt_tmpl = Template(self.config["prompt"][prompt_key])
        base = {
            "info": self.info,
            "setting": self.setting,
            "talk_history": self.talk_history,
            "whisper_history": self.whisper_history,
            "role": self.role,
            "sent_talk_count": self.sent_talk_count,
            "sent_whisper_count": self.sent_whisper_count,
            "agent_name": getattr(self, "agent_name", None),
            "game_id": getattr(self.info, "game_id", None) if self.info else None,
        }
        if extra_vars:
            base.update(extra_vars)

        prompt = prompt_tmpl.render(**base).strip()

        # Apply sleep time if configured
        if float(self.config.get("llm", {}).get("sleep_time", 0)) > 0:
            sleep(float(self.config["llm"]["sleep_time"]))

        try:
            msg = HumanMessage(content=prompt)
            if use_shared_history:
                history = self.llm_message_history + [msg]
                response = (self.llm_model | StrOutputParser()).invoke(history)
                # 共有する場合のみ履歴を汚染
                self.llm_message_history.append(msg)
                self.llm_message_history.append(AIMessage(content=response))
            else:
                # 単発呼び出し
                response = (self.llm_model | StrOutputParser()).invoke([msg])

            model_info = f"{type(self.llm_model).__name__}"
            self.agent_logger.llm_interaction(log_tag or prompt_key, prompt, response, model_info)
            return response
        except Exception as e:
            self.agent_logger.llm_error(log_tag or prompt_key, str(e), prompt)
            self.agent_logger.logger.exception("send_message_to_llm failed")
            return None

    # =========================
    # 追加: 安全化 I/O のための小ユーティリティ
    # =========================

    def _render_prompt(self, prompt_key: str, extra_vars: dict[str, Any] | None = None) -> str | None:
        """config['prompt'][prompt_key] を Jinja2 でレンダリングし、文字列を返す"""
        if "prompt" not in self.config or prompt_key not in self.config["prompt"]:
            self.agent_logger.logger.error(f"Prompt key not found: {prompt_key}")
            return None
        prompt_tmpl = Template(self.config["prompt"][prompt_key])
        base = {
            "info": self.info,
            "setting": self.setting,
            "talk_history": self.talk_history,
            "whisper_history": self.whisper_history,
            "role": self.role,
            "sent_talk_count": self.sent_talk_count,
            "sent_whisper_count": self.sent_whisper_count,
            "agent_name": getattr(self, "agent_name", None),
            "game_id": getattr(self.info, "game_id", None) if self.info else None,
        }
        if extra_vars:
            base.update(extra_vars)
        return prompt_tmpl.render(**base).strip()

    def _build_llm_call(self, *, tag: str, use_shared_history: bool) -> callable:
        """LLMSafeIO に渡す (prompt:str)->str の呼び出しを生成（ログ込み）"""
        def _call(prompt: str) -> str:
            if self.llm_model is None:
                self.agent_logger.logger.error("LLM is not initialized")
                return ""
            # 設定のスリープ（レート制限対策）
            sleep_time = float(self.config.get("llm", {}).get("sleep_time", 0) or 0)
            if sleep_time > 0:
                sleep(sleep_time)

            msg = HumanMessage(content=prompt)
            try:
                if use_shared_history:
                    history = self.llm_message_history + [msg]
                    response = (self.llm_model | StrOutputParser()).invoke(history)
                    # 履歴に追加
                    self.llm_message_history.append(msg)
                    self.llm_message_history.append(AIMessage(content=response))
                else:
                    response = (self.llm_model | StrOutputParser()).invoke([msg])

                model_info = f"{type(self.llm_model).__name__}"
                self.agent_logger.llm_interaction(tag, prompt, response, model_info)
                return response
            except Exception as e:
                self.agent_logger.llm_error(tag, str(e), prompt)
                self.agent_logger.logger.exception(f"Safe llm_call failed: {tag}")
                return ""
        return _call

    # =========================
    # 初期化 / ライフサイクル
    # =========================

    @timeout
    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.
        """
        return self.agent_name

    def initialize(self) -> None:
        """Perform initialization for game start request.

        ゲーム開始リクエストに対する初期化処理を行う.
        """
        if self.info is None:
            return

        model_type = str(self.config["llm"]["type"])
        match model_type:
            case "openai":
                self.llm_model = ChatOpenAI(
                    model=str(self.config["openai"]["model"]),
                    temperature=float(self.config["openai"]["temperature"]),
                    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                )
            case "google":
                self.llm_model = ChatGoogleGenerativeAI(
                    model=str(self.config["google"]["model"]),
                    temperature=float(self.config["google"]["temperature"]),
                    google_api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
                )
            case "ollama":
                self.llm_model = ChatOllama(
                    model=str(self.config["ollama"]["model"]),
                    temperature=float(self.config["ollama"]["temperature"]),
                    base_url=str(self.config["ollama"]["base_url"]),
                )
            case _:
                raise ValueError(model_type, "Unknown LLM type")

        # 初回応答（initialize用）
        self._send_message_to_llm(self.request)

        if self.info:
            try:
                self._ensure_macro_assets()
            except Exception:
                self.agent_logger.logger.exception("Failed to ensure macro assets")

        # AnalysisTrackerの初期化
        try:
            # info.agentのカタカナ名を使用（利用可能な場合）、そうでなければself.agent_nameを使用
            analysis_agent_name = self.info.agent if self.info and hasattr(self.info, 'agent') else self.agent_name
            self.analysis_tracker = AnalysisTracker(
                config=self.config,
                agent_name=analysis_agent_name,
                game_id=self.info.game_id if self.info else "",
                agent_logger=self.agent_logger,
                agent_obj=self
            )
            self.agent_logger.logger.info(f"AnalysisTracker initialized successfully with agent_name: {analysis_agent_name}")
        except Exception as e:
            self.agent_logger.logger.error(f"Failed to initialize AnalysisTracker: {e}")

        # Talk history initialization for all peers
        if self.info and self.info.status_map:
            peers = list(self.info.status_map.keys())
        else:
            peers = []  # フォールバック（極端なケース）
        try:
            base_dir = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi")
            init_talk_history_for_agent(
                base_dir=base_dir,
                game_id=self.info.game_id,
                agent=self.info.agent if hasattr(self.info, "agent") else self.agent_name,
                peers=peers,
                default_score=0.5,
                overwrite=False,
                logger_obj=self.agent_logger.logger,
            )
            self.agent_logger.logger.info(f"Talk history files initialized for {len(peers)} peers")
        except Exception:
            self.agent_logger.logger.exception("Failed to init talk_history files")

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        """
        self._send_message_to_llm(self.request)

        # AnalysisTrackerでトーク履歴を分析
        if self.analysis_tracker and self.info:
            try:
                # 現在のリクエストカウントを取得（daily_initializeはdayを使用）
                request_count = self.info.day if hasattr(self.info, 'day') else 0
                added = self.analysis_tracker.analyze_talk(
                    talk_history=self.talk_history,
                    info=self.info,
                    request_count=request_count
                )
                # 追加0件でも必ず保存（空ファイル生成を保証）
                try:
                    self.analysis_tracker.save_analysis()
                    # 同期処理でトレースファイルとの整合性を保証
                    self.analysis_tracker.sync_with_trace()
                    self._update_talk_logs()          # トーク履歴の更新
                    self._update_affinity_trust()     # liking/creditabilityスコアの更新
                    self._apply_affinity_to_analysis()# 信頼度調整の適用
                    self._update_self_talk_log()      # ← 追加: 自分の発話の蓄積
                    self._update_micro_belief()       # micro_belief.yml 生成/更新
                    self._generate_micro_desire(trigger="daily_initialize")  # micro_desire生成
                    self._generate_micro_intention(trigger="daily_initialize")  # micro_intention生成
                    self.agent_logger.logger.info(f"Analysis saved for day {request_count} (added={added})")
                except Exception as e:
                    self.agent_logger.logger.exception(f"Post-daily-initialize save failed: {e}")
            except Exception:
                self.agent_logger.logger.exception("Failed to analyze talk history")

    # =========================
    # TALK / WHISPER（安全化）
    # =========================

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.
        """
        # whisper 用プロンプトをレンダリングして安全化 I/O で実行
        prompt_key = "whisper"
        prompt = self._render_prompt(prompt_key)
        if not prompt:
            # フォールバック：従来関数（最終的には safe 層で 1 行化されるため非推奨）
            response = self._send_message_to_llm(self.request)
            self.sent_whisper_count = len(self.whisper_history)
            return (response or "").strip()

        llm_call = self._build_llm_call(tag="whisper_safe", use_shared_history=True)
        safe_io = LLMSafeIO(llm_call=llm_call, max_retries=1)
        allowed = self.get_alive_agents()
        response = safe_io.safe_one_line_talk(
            prompt,
            allowed_names=allowed,
            max_total_len=125,
            ascii_only=False,  # 日本語を許容
        )
        self.sent_whisper_count = len(self.whisper_history)
        return response or ""

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.
        """
        # ==== micro_intention-aware talk ====
        def _load_yaml_safe_local(p: Path) -> dict[str, Any]:
            try:
                if p.exists():
                    with p.open("r", encoding="utf-8") as f:
                        return yaml.safe_load(f) or {}
            except Exception:
                self.agent_logger.logger.exception(f"YAML load failed: {p}")
            return {}

        def _get_latest_micro_intention_entry(game_id_: str, agent_id_: str) -> dict[str, Any] | None:
            base_micro = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi")
            mi_path = base_micro / game_id_ / agent_id_ / "micro_intention.yml"
            data = _load_yaml_safe_local(mi_path)
            # 新形式（append-only）
            if isinstance(data.get("micro_intentions"), list) and data["micro_intentions"]:
                entry = data["micro_intentions"][-1]
                if isinstance(entry, dict) and "content" in entry:
                    return {"consist": str(entry.get("consist", "")).strip(),
                            "content": str(entry.get("content", "")).strip()}
            # 旧形式（単体オブジェクト）に緩く対応
            if isinstance(data.get("micro_intention"), dict):
                mi = data["micro_intention"]
                if "content" in mi and "consist" in mi:
                    return {"consist": str(mi.get("consist", "")).strip(),
                            "content": str(mi.get("content", "")).strip()}
            return None

        def _get_behavior_tendency_map(game_id_: str, agent_id_: str) -> dict[str, Any]:
            base_macro = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/macro_bdi")
            mb_path = base_macro / game_id_ / agent_id_ / "macro_belief.yml"
            mb = _load_yaml_safe_local(mb_path)
            beh = {}
            try:
                if "macro_belief" in mb:
                    beh = mb["macro_belief"].get("behavior_tendency", {}) or {}
                    if isinstance(beh, dict) and "behavior_tendencies" in beh:
                        beh = beh["behavior_tendencies"]
            except Exception:
                pass
            return beh or {}

        # micro_intention / behavior_tendency の収集
        game_id = self.info.game_id if self.info else ""
        agent_id = self.info.agent if (self.info and hasattr(self.info, "agent")) else self.agent_name
        mi_entry = _get_latest_micro_intention_entry(game_id, agent_id)
        beh_map = _get_behavior_tendency_map(game_id, agent_id)

        # ---- プロンプトをレンダリング
        if mi_entry:
            prompt = self._render_prompt(
                "talk",
                extra_vars={
                    "micro_intention_entry": mi_entry,
                    "behavior_tendency": beh_map,
                    "char_limits": {"mention_length": 125, "base_length": 125},
                },
            )
            tag = "talk_safe"
        else:
            prompt = self._render_prompt("talk_fallback")
            tag = "talk_fallback_safe"

        if not prompt:
            # フォールバック：旧実装（サニタイズは safe 層ではなく簡易に）
            response_raw = self.send_message_to_llm(
                "talk_fallback" if not mi_entry else "talk",
                log_tag="talk_legacy_fallback",
                use_shared_history=True,
            ) or ""
            # 最低限の 1 行化と禁止記号除去のみ
            response = re.sub(r"\s+", " ", response_raw).replace("`", "").replace(">", " ").replace(",", " ").strip()
            self.sent_talk_count = len(self.talk_history)
            return response[:125]

        # ---- 安全化 I/O で 1 行出力を強制
        llm_call = self._build_llm_call(tag=tag, use_shared_history=True)
        safe_io = LLMSafeIO(llm_call=llm_call, max_retries=1)
        allowed = self.get_alive_agents()
        response = safe_io.safe_one_line_talk(
            prompt,
            allowed_names=allowed,
            max_total_len=125,
            ascii_only=False,  # 日本語を許容
        )

        self.sent_talk_count = len(self.talk_history)

        # ---- AnalysisTrackerでトーク履歴を分析（talkリクエスト毎に実行）
        if self.analysis_tracker and self.info:
            try:
                # 現在のリクエストカウントを取得
                request_count = len([h for h in self.talk_history if h.agent == self.agent_name])
                added = self.analysis_tracker.analyze_talk(
                    talk_history=self.talk_history,
                    info=self.info,
                    request_count=request_count
                )
                # 追加0件でも必ず保存（空ファイル生成を保証）
                try:
                    self.analysis_tracker.save_analysis()
                    # 同期処理でトレースファイルとの整合性を保証
                    self.analysis_tracker.sync_with_trace()
                    self._update_talk_logs()          # トーク履歴の更新
                    self._update_affinity_trust()     # liking/creditabilityスコアの更新
                    self._apply_affinity_to_analysis()# 信頼度調整の適用
                    self._update_self_talk_log()      # ← 追加: 自分の発話の蓄積
                    self._update_micro_belief()       # micro_belief.yml 生成/更新
                    self._generate_micro_desire(trigger="talk")  # micro_desire生成
                    self._generate_micro_intention(trigger="talk")  # micro_intention生成
                    self.agent_logger.logger.info(f"Talk analysis saved (added={added})")
                except Exception as e:
                    self.agent_logger.logger.exception(f"Post-talk save failed: {e}")
            except Exception:
                self.agent_logger.logger.exception("Failed to analyze talk in talk()")
                # Fallback: Generate micro_desire and micro_intention anyway
                try:
                    self._generate_micro_desire(trigger="talk_fallback")
                    self._generate_micro_intention(trigger="talk_fallback")
                except Exception:
                    self.agent_logger.logger.exception("Fallback micro generation failed")

        return response or ""

    # =========================
    # デイリー終了
    # =========================

    def daily_finish(self) -> None:
        """Perform processing for daily finish request.

        昼終了リクエストに対する処理を行う.
        """
        self._send_message_to_llm(self.request)

        # AnalysisTrackerで最終的なトーク分析と保存
        if self.analysis_tracker and self.info:
            try:
                # 現在のリクエストカウントを取得
                request_count = self.info.day if hasattr(self.info, 'day') else 0
                added = self.analysis_tracker.analyze_talk(
                    talk_history=self.talk_history,
                    info=self.info,
                    request_count=request_count
                )
                # 追加0件でも必ず保存（空ファイル生成を保証）
                try:
                    self.analysis_tracker.save_analysis()
                    # 同期処理でトレースファイルとの整合性を保証
                    self.analysis_tracker.sync_with_trace()
                    self._update_talk_logs()          # トーク履歴の更新
                    self._update_affinity_trust()     # liking/creditabilityスコアの更新
                    self._apply_affinity_to_analysis()# 信頼度調整の適用
                    self._update_self_talk_log()      # ← 追加: 自分の発話の蓄積
                    self._update_micro_belief()       # micro_belief.yml 生成/更新
                    self._generate_micro_desire(trigger="daily_finish")  # micro_desire生成
                    self._generate_micro_intention(trigger="daily_finish")  # micro_intention生成
                    self.agent_logger.logger.info(f"Final analysis saved for day {request_count} (added={added})")
                except Exception as e:
                    self.agent_logger.logger.exception(f"Post-daily-finish save failed: {e}")
            except Exception:
                self.agent_logger.logger.exception("Failed to save final analysis")

    # =========================
    # 役職アクション / 投票（安全化）
    # =========================

    def _safe_target_from_prompt(self, *, prompt_key: str, tag: str) -> str:
        """指定プロンプトから有効な生存者名のみを抽出して返す（フェイルセーフ内蔵）"""
        prompt = self._render_prompt(prompt_key)
        alive = self.get_alive_agents()
        if not prompt:
            # 何らかの理由でプロンプトが無ければ最初の生存者 or ランダム
            return alive[0] if alive else ""

        llm_call = self._build_llm_call(tag=tag, use_shared_history=False)
        safe_io = LLMSafeIO(llm_call=llm_call, max_retries=1)
        target = safe_io.safe_target_only(prompt, alive_agents=alive)
        if target:
            return target
        # 念のためのフォールバック
        return random.choice(alive) if alive else ""  # noqa: S311

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.
        """
        return self._safe_target_from_prompt(prompt_key="divine", tag="divine_safe")

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.
        """
        return self._safe_target_from_prompt(prompt_key="guard", tag="guard_safe")

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.
        """
        return self._safe_target_from_prompt(prompt_key="vote", tag="vote_safe")

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.
        """
        return self._safe_target_from_prompt(prompt_key="attack", tag="attack_safe")

    # =========================
    # 終了処理
    # =========================

    def finish(self) -> None:
        """Perform processing for game finish request.

        ゲーム終了リクエストに対する処理を行う.
        """
        # ゲーム終了時に最終的な分析を保存
        if self.analysis_tracker and self.info:
            try:
                # 最終的なトーク分析と保存
                added = self.analysis_tracker.analyze_talk(
                    talk_history=self.talk_history,
                    info=self.info,
                    request_count=999  # ゲーム終了時の特別なリクエストカウント
                )
                # 追加0件でも必ず保存（空ファイル生成を保証）
                try:
                    self.analysis_tracker.save_analysis()
                    # 同期処理でトレースファイルとの整合性を保証
                    self.analysis_tracker.sync_with_trace()
                    self._update_talk_logs()          # トーク履歴の更新
                    self._update_affinity_trust()     # liking/creditabilityスコアの更新
                    self._apply_affinity_to_analysis()# 信頼度調整の適用
                    self._update_self_talk_log()      # ← 追加: 自分の発話の蓄積
                    self._update_micro_belief()       # micro_belief.yml 生成/更新
                    self._generate_micro_desire(trigger="finish")  # micro_desire生成
                    self._generate_micro_intention(trigger="finish")  # micro_intention生成
                    self.agent_logger.logger.info(f"Game finish analysis saved (added={added})")
                except Exception as e:
                    self.agent_logger.logger.exception(f"Post-game-finish save failed: {e}")
            except Exception:
                self.agent_logger.logger.exception("Failed to save game finish analysis")

    # =========================
    # BDI & 補助ユーティリティ
    # =========================

    def _update_talk_logs(self) -> None:
        """analysis.yml(またはanalysis_test.yml)からfrom別にcontent+creditabilityを抽出し、トーク履歴/<from>.yml に追記する"""
        if not self.info:
            return
        try:
            base_dir = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi")
            agent_id = self.info.agent if hasattr(self.info, "agent") else self.agent_name
            agent_dir = base_dir / self.info.game_id / agent_id
            talk_dir = determine_talk_dir(agent_dir)
            self.agent_logger.logger.info(f"Using talk directory: {talk_dir.name}")

            extract_pairs_for_agent(
                base_dir=base_dir,
                game_id=self.info.game_id,
                agent=agent_id,
                out_subdir=talk_dir.name,
                skip_pending=False,   # pending も含める（必要なら True に）
                flat_output=True,     # トーク履歴ディレクトリ直下に出力
                exclude_self=True,    # 自分の from は除外
                append=True,          # 追記モード
            )
        except Exception:
            self.agent_logger.logger.exception("Failed to update talk logs")

    def _update_affinity_trust(self) -> None:
        """トーク履歴ファイルにliking/creditabilityスコアを更新する"""
        if not self.info:
            return
        try:
            base_dir = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi")
            agent_id = self.info.agent if hasattr(self.info, "agent") else self.agent_name
            agent_dir = base_dir / self.info.game_id / agent_id
            talk_dir = determine_talk_dir(agent_dir)

            update_affinity_trust_for_agent(
                config=self.config,
                game_id=self.info.game_id,
                agent=agent_id,
                agent_logger=self.agent_logger,
                talk_dir_name=talk_dir.name,
                agent_obj=self
            )
        except Exception:
            self.agent_logger.logger.exception("Failed to update affinity/trust headers in talk history")

    def _apply_affinity_to_analysis(self) -> None:
        """analysis.ymlのcredibilityスコアにliking/creditabilityを係数として適用する"""
        if not self.info:
            return
        try:
            base_dir = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi")
            apply_affinity_to_analysis(
                base_dir=base_dir,
                game_id=self.info.game_id,
                agent=self.info.agent if hasattr(self.info, "agent") else self.agent_name,
                logger_obj=self.agent_logger.logger
            )
        except Exception:
            self.agent_logger.logger.exception("Failed to apply affinity to analysis")

    def _update_self_talk_log(self) -> None:
        """Agent.talk_history から自分の発話のみを self_talk.yml に追記する"""
        if not self.info:
            return
        try:
            agent_id = self.info.agent if hasattr(self.info, "agent") else self.agent_name
            path = write_self_talk_for_agent(
                game_id=self.info.game_id,
                agent=agent_id,
                talks=self.talk_history,
            )
            self.agent_logger.logger.info(f"self_talk saved: {path}")
        except Exception:
            self.agent_logger.logger.exception("Failed to update self_talk log")

    def _update_micro_belief(self) -> None:
        """analysis.yml と talk_history をもとに micro_belief.yml を生成/更新する"""
        if not self.info:
            return
        try:
            path = write_micro_belief(
                game_id=self.info.game_id,
                owner=self.info.agent if hasattr(self.info, "agent") else self.agent_name
            )
            self.agent_logger.logger.info(f"micro_belief saved: {path}")
        except Exception:
            self.agent_logger.logger.exception("Failed to generate micro_belief")

    def _generate_micro_desire(self, *, trigger: str) -> None:
        """Generate micro_desire based on current situation."""
        if not self.info:
            return
        try:
            path = generate_micro_desire_for_agent(
                game_id=self.info.game_id,
                agent=self.info.agent if hasattr(self.info, "agent") else self.agent_name,
                agent_obj=self,
                logger_obj=self.agent_logger,
                trigger=trigger
            )
            if path:
                self.agent_logger.logger.info(f"micro_desire saved: {path}")
        except Exception:
            self.agent_logger.logger.exception("Failed to generate micro_desire")

    def _generate_micro_intention(self, *, trigger: str) -> None:
        """Generate micro_intention based on micro_desire."""
        if not self.info:
            return
        try:
            path = generate_micro_intention_for_agent(
                game_id=self.info.game_id,
                agent=self.info.agent if hasattr(self.info, "agent") else self.agent_name,
                agent_obj=self,
                logger_obj=self.agent_logger,
                trigger=trigger
            )
            if path:
                self.agent_logger.logger.info(f"micro_intention saved: {path}")
        except Exception:
            self.agent_logger.logger.exception("Failed to generate micro_intention")

    def _ensure_macro_assets(self) -> None:
        """Ensure macro_belief and macro_desire files exist (macro_plan disabled)."""
        if not self.info:
            return

        base = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025")
        macro_dir = base / "info/bdi_info/macro_bdi" / self.info.game_id / self.info.agent

        mb_path = macro_dir / "macro_belief.yml"
        md_path = macro_dir / "macro_desire.yml"
        # mp_path = macro_dir / "macro_plan.yml"  # [macro_plan disabled]

        # Get role information
        role_en = None
        if hasattr(self.info, 'role_map') and self.info.agent in self.info.role_map:
            role_obj = self.info.role_map[self.info.agent]
            role_en = role_obj.name if hasattr(role_obj, 'name') else str(role_obj)
        elif self.role:
            role_en = self.role.name if hasattr(self.role, 'name') else str(self.role)

        # Determine model for LLM calls
        model = ((self.config.get("openai", {}) or {}).get("model") or
                 (self.config.get("google", {}) or {}).get("model") or
                 "gpt-4o")

        # 1. Ensure macro_belief.yml exists
        if not mb_path.exists():
            try:
                self.agent_logger.logger.info("Generating macro_belief.yml")
                macro_belief_path = generate_macro_belief(
                    config=self.config,
                    profile=(self.info.profile or ""),
                    agent_name=self.info.agent,
                    game_id=self.info.game_id,
                    role_en=role_en,
                    agent_logger=self.agent_logger,
                    agent_obj=self
                )
                self.agent_logger.logger.info(f"Generated macro belief: {macro_belief_path}")
            except Exception:
                self.agent_logger.logger.exception("macro_belief generation failed")
                # Continue to next step even if this fails

        # 2. Ensure macro_desire.yml exists
        if not md_path.exists():
            try:
                self.agent_logger.logger.info("Generating macro_desire.yml")
                macro_desire_data = generate_macro_desire(
                    game_id=self.info.game_id,
                    agent=self.info.agent,
                    agent_obj=self,
                    dry_run=False,
                    overwrite=False
                )
                self.agent_logger.logger.info(f"Generated macro desire for agent: {self.info.agent}")
            except Exception as e:
                self.agent_logger.logger.exception("macro_desire generation failed")
                # Write fallback minimal YAML
                try:
                    self._write_fallback_macro_desire(md_path, str(e))
                except Exception:
                    self.agent_logger.logger.exception("Failed to write fallback macro_desire")

        # 3. macro_plan generation: DISABLED

    # def _ensure_deferred_macro_plan(self) -> None:
    #     """Ensure macro_plan.yml exists - DISABLED."""
    #     return

    def _write_fallback_macro_desire(self, file_path: Path, error_reason: str) -> None:
        """Write minimal fallback macro_desire.yml structure."""
        from datetime import datetime
        import yaml

        file_path.parent.mkdir(parents=True, exist_ok=True)

        fallback_data = {
            "macro_desire": {
                "summary": "Auto-generated (fallback)",
                "description": f"Fallback due to error: {error_reason[:100]}"
            },
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "source": "agent.py fallback",
                "agent": self.info.agent if self.info else self.agent_name,
                "game_id": self.info.game_id if self.info else "",
                "fallback": True,
                "error": error_reason[:200]
            }
        }

        with file_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(fallback_data, f, allow_unicode=True, sort_keys=False)

        self.agent_logger.logger.info(f"Created fallback macro_desire: {file_path}")

    # =========================
    # ディスパッチ
    # =========================

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.
        """
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None
