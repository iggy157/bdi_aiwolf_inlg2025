# src/llm_safe_agent_io.py
from __future__ import annotations

import re
from typing import Callable, Sequence


class LLMSafeIO:
    """Minimal guardrail layer for AIWolf agents.

    - safe_one_line_talk: LLMの出力を1行・記号除去・長さ制限・（先頭メンションがあれば）候補内に矯正
    - safe_target_only  : 候補の中から1名のみを厳格に選択（名前文字列だけを返す）
    """

    def __init__(self, llm_call: Callable[[str], str], max_retries: int = 1) -> None:
        self.llm_call = llm_call
        self.max_retries = max(0, int(max_retries))

    # --------------------------------------------------------------------- #
    # Public APIs
    # --------------------------------------------------------------------- #
    def safe_one_line_talk(
        self,
        prompt: str,
        allowed_names: Sequence[str] | None = None,
        *,
        max_total_len: int = 125,
        ascii_only: bool = False,
    ) -> str:
        """プロンプトを実行し、1行の安全な発話に整形して返す。"""
        constraint = (
            "\n\n[OUTPUT FORMAT RULES]\n"
            "- Return ONE line only. No line breaks, no lists, no markdown.\n"
            f"- Keep within {max_total_len} characters.\n"
            "- If you mention another player at the start, use the exact name from the candidate list if provided.\n"
            "- Do not include quotes, backticks, angle brackets, or code fences.\n"
        )
        if allowed_names:
            constraint += "- Candidate names: " + ", ".join(str(n) for n in allowed_names) + "\n"

        raw = self.llm_call(prompt + constraint)
        text = self._postprocess_line(raw, max_total_len=max_total_len, ascii_only=ascii_only)

        # 先頭メンションがあれば許可リストに矯正（不一致ならメンションを除去）
        if allowed_names and text.startswith("@"):
            mention_token = self._extract_leading_mention(text)
            if mention_token:
                best = self._best_match_name(mention_token, allowed_names)
                if best:
                    text = "@" + best + " " + text[len(mention_token) + 1 :].lstrip(":： \t")
                else:
                    # 無効メンションは削除
                    text = text[len(mention_token) + 1 :].lstrip(":： \t")

        return self._truncate(text, max_total_len)

    def safe_target_only(self, prompt: str, *, alive_agents: Sequence[str]) -> str:
        """候補（alive_agents）からちょうど1名を選び、その“名前のみ”返す。"""
        if not alive_agents:
            return ""
        name_line = " / ".join(alive_agents)
        forced = (
            f"{prompt}\n\n"
            "[TASK]\n"
            "上の依頼に基づいて、次の候補から**ちょうど1つ**の名前だけを選び、\n"
            "候補にある表記を**そのまま**出力してください。\n"
            f"CANDIDATES: {name_line}\n"
            "出力は名前のみ。他の語・記号・説明は禁止。\n"
        )
        raw = (self.llm_call(forced) or "").strip()
        candidate = self._extract_name_from_text(raw, alive_agents)
        if candidate:
            return candidate

        # 失敗時はさらに厳格な英語プロンプトで再試行
        for _ in range(self.max_retries):
            forced2 = (
                "Choose exactly ONE from the following names and output ONLY the chosen name with exact spelling.\n"
                f"OPTIONS: {name_line}\n"
                "Answer with the name only.\n"
            )
            raw2 = (self.llm_call(forced2) or "").strip()
            candidate = self._extract_name_from_text(raw2, alive_agents)
            if candidate:
                return candidate

        return ""

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _postprocess_line(self, s: str, *, max_total_len: int, ascii_only: bool) -> str:
        s = s or ""
        s = s.replace("\n", " ").replace("\r", " ")
        # 箇条書き・引用記号の除去（先頭）
        s = re.sub(r"^\s*[-*•>]+\s*", "", s)
        # コード記法や危険記号の除去
        s = s.replace("```", " ").replace("`", "").replace(">", " ").replace(",", " ")
        # 余分な空白の圧縮
        s = " ".join(s.split())
        # 絵文字・特殊記号の削除（緩め）
        s = re.sub(r"[\u2600-\u27BF\uE000-\uF8FF\U0001F300-\U0001FAFF]", "", s)
        if ascii_only:
            s = s.encode("ascii", errors="ignore").decode("ascii")
        return self._truncate(s.strip(), max_total_len)

    def _truncate(self, s: str, n: int) -> str:
        return s if len(s) <= n else s[:n]

    def _normalize(self, name: str) -> str:
        # 括弧や日本語名にもそこそこ耐える緩めの正規化
        return re.sub(r"[^0-9a-zA-Zぁ-んァ-ヶ一-龠\[\]_]+", "", name).lower()

    def _extract_leading_mention(self, s: str) -> str | None:
        # 例: "@Agent[02]: こんにちは"
        m = re.match(r"^@([^\s:：]+)", s)
        return m.group(1) if m else None

    def _best_match_name(self, token: str, names: Sequence[str]) -> str | None:
        if not token:
            return None
        tok = self._normalize(token)
        best: str | None = None
        for nm in names:
            nmn = self._normalize(nm)
            if tok == nmn:
                return nm
            # 緩い部分一致（片方に含まれていればOK）
            if tok in nmn or nmn in tok:
                best = best or nm
        return best

    def _extract_name_from_text(self, text: str, names: Sequence[str]) -> str | None:
        if not text:
            return None
        t = text.strip()

        # 完全一致 or クォート付き一致
        for nm in names:
            if t == nm or t.strip("\"'` ") == nm:
                return nm

        # JSON風: {"target":"Agent[03]"} / target=Agent[03]
        m = re.search(r'target["\']?\s*[:=]\s*["\']?([^"\'\s}]+)', t, flags=re.I)
        if m:
            cand = m.group(1)
            mm = self._best_match_name(cand, names)
            if mm:
                return mm

        # 先頭トークン or 先頭メンション
        m2 = re.match(r"^@?([^\s,;:：，、]+)", t)
        if m2:
            cand = m2.group(1)
            mm = self._best_match_name(cand, names)
            if mm:
                return mm

        # 全文からの緩い部分一致
        low = t.lower()
        for nm in names:
            if nm.lower() in low:
                return nm

        return None
