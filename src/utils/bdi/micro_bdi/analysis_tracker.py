"""analysis.ymlを生成するためのモジュール."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
        # {packet_idx: [{"content": ..., "type": ..., "from": ..., "to": ..., "request_count": ..., "credibility": ...}, ...]}
        self.analysis_history: dict[int, list[dict[str, Any]]] = {}
        
        # 前回分析したトークの数を記録
        self.last_analyzed_talk_count = 0
        
        # リクエストカウンターを初期化
        self.request_count = 0
        
        # LLMクライアントの初期化
        self.llm_model = None
        self._initialize_llm()
        
        # 出力ディレクトリの設定
        self._setup_output_directory()
    
    def _initialize_llm(self) -> None:
        """LLMクライアントを初期化."""
        from pathlib import Path
        load_dotenv(Path(__file__).parent.joinpath("./../../../../config/.env"))
        
        model_type = str(self.config.get("llm", {}).get("type", "openai"))
        try:
            match model_type:
                case "openai":
                    self.llm_model = ChatOpenAI(
                        model=str(self.config.get("openai", {}).get("model", "gpt-4o")),
                        temperature=float(self.config.get("openai", {}).get("temperature", 0.7)),
                        api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "")),
                    )
                case "google":
                    self.llm_model = ChatGoogleGenerativeAI(
                        model=str(self.config.get("google", {}).get("model", "gemini-2.0-flash-lite")),
                        temperature=float(self.config.get("google", {}).get("temperature", 0.7)),
                        api_key=SecretStr(os.environ.get("GOOGLE_API_KEY", "")),
                    )
                case "ollama":
                    self.llm_model = ChatOllama(
                        model=str(self.config.get("ollama", {}).get("model", "llama3.1")),
                        temperature=float(self.config.get("ollama", {}).get("temperature", 0.7)),
                        base_url=str(self.config.get("ollama", {}).get("base_url", "http://localhost:11434")),
                    )
                case _:
                    print(f"Unknown LLM type: {model_type}")
                    self.llm_model = None
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            self.llm_model = None
    
    def _setup_output_directory(self) -> None:
        """出力ディレクトリの設定."""
        # /info/bdi_info/micro_bdi/game_id/agent_name/ の形式でディレクトリを作成
        self.output_dir = Path("info") / "bdi_info" / "micro_bdi" / self.game_id / self.agent_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_file = self.output_dir / "analysis.yml"
    
    def analyze_talk(
        self,
        talk_history: list[Talk],
        info: Info,
        request_count: int | None = None,
    ) -> None:
        """トーク履歴を分析してanalysis.ymlを更新."""
        # 新しいトーク履歴のエントリだけを分析
        new_talks = talk_history[self.last_analyzed_talk_count:]
        
        print(f"[AnalysisTracker] Analyzing talk for {self.agent_name}")
        print(f"[AnalysisTracker] Total talk_history: {len(talk_history)}")
        print(f"[AnalysisTracker] New talks to analyze: {len(new_talks)}")
        print(f"[AnalysisTracker] Last analyzed count: {self.last_analyzed_talk_count}")
        
        if not new_talks:
            print(f"[AnalysisTracker] No new talks to analyze")
            return  # 新しいトークがない場合は何もしない
        
        # request_countが渡された場合は更新
        if request_count is not None:
            self.request_count = request_count
        
        self.packet_idx += 1
        
        # 新しいトーク履歴のエントリを分析
        analysis_entries = []
        
        for i, talk in enumerate(new_talks):
            print(f"[AnalysisTracker] Analyzing talk {i+1}/{len(new_talks)}: {talk.agent} - {talk.text}")
            # 発話内容の分析
            analysis_entry = self._analyze_talk_entry(talk, info)
            if analysis_entry:
                analysis_entries.append(analysis_entry)
                print(f"[AnalysisTracker] Analysis entry created: {analysis_entry['type']} from {analysis_entry['from']}")
            else:
                print(f"[AnalysisTracker] No analysis entry created for talk: {talk.text}")
        
        # 分析結果を履歴に保存
        if analysis_entries:
            self.analysis_history[self.packet_idx] = analysis_entries
            print(f"[AnalysisTracker] Saved {len(analysis_entries)} analysis entries for packet_idx {self.packet_idx}")
        else:
            print(f"[AnalysisTracker] No analysis entries to save")
        
        # 解析直後の詳細ログ（デバッグ用）
        print(f"[AnalysisTracker] === 解析直後のanalysis_history状態 ===")
        print(f"[AnalysisTracker] Total analysis_history keys: {len(self.analysis_history)}")
        total_entries = 0
        for key, entries in self.analysis_history.items():
            entry_count = len(entries) if entries else 0
            total_entries += entry_count
            print(f"[AnalysisTracker]   packet_idx {key}: {entry_count} entries")
            # 各エントリの詳細も表示
            for i, entry in enumerate(entries[:3]):  # 最初の3件のみ表示
                print(f"[AnalysisTracker]     [{i}] {entry.get('from', 'N/A')}: '{entry.get('content', 'N/A')[:50]}{'...' if len(str(entry.get('content', ''))) > 50 else ''}' -> type: {entry.get('type', 'N/A')}")
            if len(entries) > 3:
                print(f"[AnalysisTracker]     ... and {len(entries) - 3} more entries")
        print(f"[AnalysisTracker] Total entries across all packets: {total_entries}")
        print(f"[AnalysisTracker] === 解析直後状態ログ終了 ===")
        
        # 分析済みトーク数を更新
        self.last_analyzed_talk_count = len(talk_history)
        
        # 一時的な保存テスト（デバッグ用）
        print(f"[AnalysisTracker] === 一時的保存テスト実行 ===")
        self.save_analysis()
        print(f"[AnalysisTracker] === 一時的保存テスト終了 ===")
    
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
                "request_count": self.request_count,
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
            "request_count": self.request_count,
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
            
            # プロンプトテンプレートの作成
            prompt_template = self.config.get("prompt", {}).get("analyze_message_type", "")
            if not prompt_template:
                if self.agent_logger:
                    self.agent_logger.llm_error("message_type_analysis", "No prompt template found for analyze_message_type")
                return "null"
            
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
            
            # プロンプトテンプレートの作成
            prompt_template = self.config.get("prompt", {}).get("analyze_target_agents", "")
            if not prompt_template:
                if self.agent_logger:
                    self.agent_logger.llm_error("target_agents_analysis", "No prompt template found for analyze_target_agents")
                return "null"
            
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
            
            # プロンプトテンプレートの作成
            prompt_template = self.config.get("prompt", {}).get("analyze_credibility", "")
            if not prompt_template:
                if self.agent_logger:
                    self.agent_logger.llm_error("credibility_analysis", "No prompt template found for analyze_credibility")
                return {}
            
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
        # request_countごとにグループ化
        request_groups: dict[int, list[dict[str, Any]]] = {}
        
        # 全エントリをrequest_countでグループ化
        for packet_idx, entries in self.analysis_history.items():
            for entry in entries:
                request_count = entry.get("request_count", 0)
                if request_count not in request_groups:
                    request_groups[request_count] = []
                request_groups[request_count].append(entry)
        
        # 保存直前の詳細ログ（デバッグ用）
        print(f"[AnalysisTracker] === 保存直前のanalysis_history状態 ===")
        print(f"[AnalysisTracker] Saving analysis for {self.agent_name} (game_id: {self.game_id})")
        print(f"[AnalysisTracker] Total analysis_history entries: {len(self.analysis_history)}")
        total_entries_before_group = 0
        for key, entries in self.analysis_history.items():
            entry_count = len(entries) if entries else 0
            total_entries_before_group += entry_count
            print(f"[AnalysisTracker]   packet_idx {key}: {entry_count} entries")
            # 各エントリの詳細も表示
            for i, entry in enumerate(entries[:2]):  # 最初の2件のみ表示
                print(f"[AnalysisTracker]     [{i}] request_count: {entry.get('request_count', 'N/A')}, from: {entry.get('from', 'N/A')}, type: {entry.get('type', 'N/A')}")
            if len(entries) > 2:
                print(f"[AnalysisTracker]     ... and {len(entries) - 2} more entries")
        print(f"[AnalysisTracker] Total entries before grouping: {total_entries_before_group}")
        
        print(f"[AnalysisTracker] Total request_groups after grouping: {len(request_groups)}")
        total_entries_after_group = 0
        for request_count, entries in request_groups.items():
            entry_count = len(entries) if entries else 0
            total_entries_after_group += entry_count
            print(f"[AnalysisTracker]   request_count {request_count}: {entry_count} entries")
        print(f"[AnalysisTracker] Total entries after grouping: {total_entries_after_group}")
        print(f"[AnalysisTracker] Output file: {self.analysis_file}")
        print(f"[AnalysisTracker] === 保存直前状態ログ終了 ===")
        
        # 出力ディレクトリが存在することを確認
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # YAMLファイルに書き込み（一時ファイルを使用して安全に書き込み）
        temp_file = self.analysis_file.with_suffix('.tmp')
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                if not request_groups:
                    # 空の場合でもファイルは作成し、コメントを追加
                    f.write("# No analysis entries found\n")
                    f.write("# This could be due to:\n")
                    f.write("#   - No talk history available\n")
                    f.write("#   - LLM initialization failed\n")
                    f.write("#   - Analysis processing errors\n")
                    f.write(f"# Agent: {self.agent_name}\n")
                    f.write(f"# Game ID: {self.game_id}\n")
                else:
                    entry_counter = 1
                    for request_count in sorted(request_groups.keys()):
                        # リクエストヘッダーを書き込み
                        f.write(f"# Request {request_count}\n")
                        
                        # そのリクエストの全エントリを書き込み
                        for i, entry in enumerate(request_groups[request_count]):
                            f.write(f"{entry_counter}:\n")
                            f.write(f"  content: \"{entry['content']}\"\n")  # Quote content to handle YAML special characters
                            f.write(f"  type: {entry['type']}\n")
                            f.write(f"  from: {entry['from']}\n")
                            f.write(f"  to: {entry['to']}\n")
                            f.write(f"  request_count: {entry.get('request_count', 0)}\n")
                            f.write(f"  credibility: {entry.get('credibility', 0.0)}\n")
                            if i < len(request_groups[request_count]) - 1:
                                f.write("\n")
                            entry_counter += 1
                        
                        f.write("\n\n")
                
                # ファイルの内容を確実にディスクに書き込む
                f.flush()
                import os
                os.fsync(f.fileno())
            
            # 一時ファイルを本ファイルに移動（原子的操作）
            temp_file.replace(self.analysis_file)
            print(f"[AnalysisTracker] Analysis saved successfully to: {self.analysis_file}")
            
            # ファイルサイズを確認
            file_size = self.analysis_file.stat().st_size
            print(f"[AnalysisTracker] File size: {file_size} bytes")
                            
        except Exception as e:
            print(f"[AnalysisTracker] Error saving analysis: {e}")
            if self.agent_logger:
                self.agent_logger.logger.error(f"Failed to save analysis file: {e}")
            
            # 一時ファイルが残っている場合は削除
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
        
        # analysis.yml保存後にselect_sentence.ymlとintention.ymlも更新
        self._update_select_sentence()
        self._update_intention()
    
    def force_save_with_status(self) -> None:
        """強制的に現在の状態をファイルに保存（デバッグ用）."""
        print(f"[AnalysisTracker] Force saving analysis status:")
        print(f"  - agent_name: {self.agent_name}")
        print(f"  - game_id: {self.game_id}")
        print(f"  - packet_idx: {self.packet_idx}")
        print(f"  - last_analyzed_talk_count: {self.last_analyzed_talk_count}")
        print(f"  - request_count: {self.request_count}")
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
