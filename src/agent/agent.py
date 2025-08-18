"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import os
import random
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

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread
from utils.bdi.micro_bdi.extract_pairs import extract_pairs_for_agent
from utils.bdi.micro_bdi.affinity_trust_updater import update_affinity_trust_for_agent
from utils.bdi.micro_bdi.talk_history_init import init_talk_history_for_agent
from utils.bdi.micro_bdi.credibility_adjuster import apply_affinity_to_analysis
from utils.bdi.macro_bdi.macro_belief import generate_macro_belief
from utils.bdi.macro_bdi.macro_desire import generate_macro_desire
from utils.bdi.macro_bdi.macro_plan import generate_macro_plan
from utils.bdi.micro_bdi.analysis_tracker import AnalysisTracker

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
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def _send_message_to_llm(self, request: Request | None) -> str | None:
        """Send message to LLM and get response.

        LLMにメッセージを送信して応答を取得する.

        Args:
            request (Request | None): The request type to process / 処理するリクエストタイプ

        Returns:
            str | None: LLM response or None if error occurred / LLMの応答またはエラー時はNone
        """
        if request is None:
            return None
        if request.lower() not in self.config["prompt"]:
            return None
        prompt = self.config["prompt"][request.lower()]
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
        template: Template = Template(prompt)
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

    @timeout
    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.

        Returns:
            str: Agent name / エージェント名
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
                    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
                )
            case "ollama":
                self.llm_model = ChatOllama(
                    model=str(self.config["ollama"]["model"]),
                    temperature=float(self.config["ollama"]["temperature"]),
                    base_url=str(self.config["ollama"]["base_url"]),
                )
            case _:
                raise ValueError(model_type, "Unknown LLM type")
        self.llm_model = self.llm_model
        self._send_message_to_llm(self.request)
        
        if self.info and self.info.profile:
            try:
                # Get role from info.role_map or use self.role as fallback
                role_en = None
                if hasattr(self.info, 'role_map') and self.info.agent in self.info.role_map:
                    role_obj = self.info.role_map[self.info.agent]
                    role_en = role_obj.name if hasattr(role_obj, 'name') else str(role_obj)
                elif self.role:
                    role_en = self.role.name if hasattr(self.role, 'name') else str(self.role)
                
                macro_belief_path = generate_macro_belief(
                    config=self.config,
                    profile=self.info.profile,
                    agent_name=self.info.agent,
                    game_id=self.info.game_id,
                    role_en=role_en,
                    agent_logger=self.agent_logger
                )
                self.agent_logger.logger.info(f"Generated macro belief: {macro_belief_path}")
                
                # Generate macro desire based on macro belief
                try:
                    macro_desire_data = generate_macro_desire(
                        game_id=self.info.game_id,
                        agent=self.info.agent,
                        model=self.config.get("openai", {}).get("model", "gpt-4o"),
                        dry_run=False,
                        overwrite=True  # Allow overwrite during initialization
                    )
                    self.agent_logger.logger.info(f"Generated macro desire for agent: {self.info.agent}")
                except Exception as e:
                    self.agent_logger.logger.error(f"Failed to generate macro desire: {e}")
                
                # Generate macro plan based on behavior tendencies
                try:
                    macro_plan_data = generate_macro_plan(
                        game_id=self.info.game_id,
                        agent=self.info.agent,
                        model=self.config.get("openai", {}).get("model", "gpt-4o"),
                        dry_run=False,
                        overwrite=True  # Allow overwrite during initialization
                    )
                    self.agent_logger.logger.info(f"Generated macro plan for agent: {self.info.agent}")
                except Exception as e:
                    self.agent_logger.logger.error(f"Failed to generate macro plan: {e}")
            except Exception as e:
                self.agent_logger.logger.error(f"Failed to generate macro belief: {e}")
        
        # AnalysisTrackerの初期化
        try:
            # info.agentのカタカナ名を使用（利用可能な場合）、そうでなければself.agent_nameを使用
            analysis_agent_name = self.info.agent if self.info and hasattr(self.info, 'agent') else self.agent_name
            self.analysis_tracker = AnalysisTracker(
                config=self.config,
                agent_name=analysis_agent_name,
                game_id=self.info.game_id if self.info else "",
                agent_logger=self.agent_logger
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
                    self._update_talk_logs()  # トーク履歴の更新
                    self._update_affinity_trust()  # liking/creditabilityスコアの更新
                    self._apply_affinity_to_analysis()  # 信頼度調整の適用
                    self.agent_logger.logger.info(f"Analysis saved for day {request_count} (added={added})")
                except Exception as e:
                    self.agent_logger.logger.exception(f"Post-daily-initialize save failed: {e}")
            except Exception:
                self.agent_logger.logger.exception("Failed to analyze talk history")

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        response = self._send_message_to_llm(self.request)
        self.sent_whisper_count = len(self.whisper_history)
        return response or ""

    def _update_talk_logs(self) -> None:
        """analysis.yml(またはanalysis_test.yml)からfrom別にcontent+creditabilityを抽出し、トーク履歴/<from>.yml に追記する"""
        if not self.info:
            return
        try:
            base_dir = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi")
            extract_pairs_for_agent(
                base_dir=base_dir,
                game_id=self.info.game_id,
                agent=self.info.agent if hasattr(self.info, "agent") else self.agent_name,
                out_subdir="トーク履歴",
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
            update_affinity_trust_for_agent(
                config=self.config,
                game_id=self.info.game_id,
                agent=self.info.agent if hasattr(self.info, "agent") else self.agent_name,
                agent_logger=self.agent_logger,
                talk_dir_name="トーク履歴"
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

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        response = self._send_message_to_llm(self.request)
        self.sent_talk_count = len(self.talk_history)
        
        # AnalysisTrackerでトーク履歴を分析（talkリクエスト毎に実行）
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
                    self._update_talk_logs()  # トーク履歴の更新
                    self._update_affinity_trust()  # liking/creditabilityスコアの更新
                    self._apply_affinity_to_analysis()  # 信頼度調整の適用
                    self.agent_logger.logger.info(f"Talk analysis saved (added={added})")
                except Exception as e:
                    self.agent_logger.logger.exception(f"Post-talk save failed: {e}")
            except Exception:
                self.agent_logger.logger.exception("Failed to analyze talk in talk()")
        
        return response or ""

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
                    self._update_talk_logs()  # トーク履歴の更新
                    self._update_affinity_trust()  # liking/creditabilityスコアの更新
                    self._apply_affinity_to_analysis()  # 信頼度調整の適用
                    self.agent_logger.logger.info(f"Final analysis saved for day {request_count} (added={added})")
                except Exception as e:
                    self.agent_logger.logger.exception(f"Post-daily-finish save failed: {e}")
            except Exception:
                self.agent_logger.logger.exception("Failed to save final analysis")

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

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
                    self._update_talk_logs()  # トーク履歴の更新
                    self._update_affinity_trust()  # liking/creditabilityスコアの更新
                    self._apply_affinity_to_analysis()  # 信頼度調整の適用
                    self.agent_logger.logger.info(f"Game finish analysis saved (added={added})")
                except Exception as e:
                    self.agent_logger.logger.exception(f"Post-game-finish save failed: {e}")
            except Exception:
                self.agent_logger.logger.exception("Failed to save game finish analysis")

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.

        Returns:
            str | None: Action result string or None / アクションの結果文字列またはNone
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
