#!/usr/bin/env python3
import argparse
import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml

BASE_INFO_DIR = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi")
DEFAULT_OUT_SUBDIR = "トーク履歴"
UNKNOWN_FROM = "unknown"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def sanitize_name(name: str) -> str:
    if not name:
        return UNKNOWN_FROM
    s = re.sub(r'[/\\:*?"<>|\0]', "", str(name)).strip()
    return s or UNKNOWN_FROM

def normalize_from(value: Any) -> str:
    if value is None:
        return UNKNOWN_FROM
    s = str(value).strip()
    if s.lower() == "null" or s == "":
        return UNKNOWN_FROM
    return sanitize_name(s)

def normalize_agent_name(agent: str) -> str:
    return sanitize_name(agent)

def extract_cred(entry: dict[str, Any]) -> float:
    for k in ("credibility", "creditability"):
        if k in entry:
            try:
                return float(entry[k])
            except (TypeError, ValueError):
                logger.debug("invalid %s: %r", k, entry[k])
    return 0.0

def is_pending(entry: dict[str, Any]) -> bool:
    if entry.get("type") == "pending_analysis":
        return True
    c = entry.get("content", "")
    return isinstance(c, str) and c.strip().startswith("[PENDING]")

def find_analysis(agent_dir: Path) -> Optional[Path]:
    for fn in ("analysis.yml", "analysis_test.yml"):
        p = agent_dir / fn
        if p.exists():
            return p
    return None

def load_yaml_dict(p: Path) -> Optional[dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
        logger.warning("Invalid YAML (expects dict) at %s: %r", p, type(data))
    except Exception as e:
        logger.warning("Failed to read %s: %s", p, e)
    return None

def load_items_list(p: Path) -> list[dict]:
    """許容する既存フォーマット:
       - list[dict(content, creditability)]
       - dict{items: [...]}
    """
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning("Failed to read %s: %s", p, e)
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return [x for x in data["items"] if isinstance(x, dict)]
    return []

def dedup_append(existing: list[dict], new_items: list[dict]) -> list[dict]:
    seen: set[tuple[str, float]] = set()
    out: list[dict] = []
    def key_of(d: dict) -> tuple[str, float]:
        return (str(d.get("content", "")), float(d.get("creditability", 0.0)))
    # preserve order: existing -> new (skip duplicates)
    for d in existing + new_items:
        k = key_of(d)
        if k in seen:
            continue
        seen.add(k)
        out.append({"content": str(d.get("content", "")),
                    "creditability": float(d.get("creditability", 0.0))})
    return out

def extract_pairs_for_agent(
    base_dir: Path,
    game_id: str,
    agent: str,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
    skip_pending: bool = False,
    exclude_self: bool = True,
    append: bool = True,
) -> dict[str, list[dict]]:
    agent_dir = base_dir / game_id / agent
    if not agent_dir.exists():
        logger.warning("Agent dir not found: %s", agent_dir)
        return {}

    analysis_file = find_analysis(agent_dir)
    if not analysis_file:
        logger.warning("No analysis file for %s/%s", game_id, agent)
        return {}

    data = load_yaml_dict(analysis_file)
    if not data:
        return {}

    logger.info("Processing %s/%s from %s", game_id, agent, analysis_file.name)

    # group by from
    grouped: dict[str, list[dict]] = {}
    # keys: numeric order first
    def sort_key(k: Any):
        s = str(k)
        return (0, int(s)) if s.isdigit() else (1, s)
    for k in sorted(data.keys(), key=sort_key):
        entry = data[k]
        if not isinstance(entry, dict):
            continue
        if skip_pending and is_pending(entry):
            continue
        content = entry.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        from_name = normalize_from(entry.get("from"))
        self_name = normalize_agent_name(agent)
        if exclude_self and from_name != UNKNOWN_FROM and from_name == self_name:
            # 自分の発話はスキップ
            continue
        credit = extract_cred(entry)
        grouped.setdefault(from_name, []).append({"content": content, "creditability": credit})

    # write per-from file: <agent_dir>/<out_subdir>/<from>.yml
    out_base = agent_dir / out_subdir
    out_base.mkdir(parents=True, exist_ok=True)

    for from_name, items in grouped.items():
        out_path = out_base / f"{from_name}.yml"
        try:
            if append:
                existing = load_items_list(out_path)
            else:
                existing = []
            merged = dedup_append(existing, items)
            with out_path.open("w", encoding="utf-8") as f:
                f.write("# generated by utils.bdi.micro_bdi.extract_pairs\n")
                yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
            logger.info("Saved %d items -> %s", len(merged), out_path)
        except Exception as e:
            logger.warning("Failed to save %s: %s", out_path, e)

    total = sum(len(v) for v in grouped.values())
    logger.info("Completed %s/%s: %d speakers, %d pairs (new batch)", game_id, agent, len(grouped), total)
    return grouped

def extract_pairs_for_game(
    base_dir: Path,
    game_id: str,
    agents: Optional[list[str]] = None,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
    skip_pending: bool = False,
    exclude_self: bool = True,
    append: bool = True,
) -> dict[str, dict[str, list[dict]]]:
    game_dir = base_dir / game_id
    if not game_dir.exists():
        logger.warning("Game dir not found: %s", game_dir)
        return {}
    if agents is None:
        try:
            agents = [d.name for d in game_dir.iterdir() if d.is_dir()]
            logger.info("Discovered %d agents in %s: %s", len(agents), game_id, agents)
        except Exception as e:
            logger.warning("Failed to list agents in %s: %s", game_dir, e)
            return {}
    results: dict[str, dict[str, list[dict]]] = {}
    for a in agents:
        try:
            results[a] = extract_pairs_for_agent(
                base_dir, game_id, a, out_subdir, skip_pending, exclude_self, append
            )
        except Exception as e:
            logger.warning("Failed to process agent %s: %s", a, e)
            results[a] = {}
    return results

def main() -> int:
    p = argparse.ArgumentParser(description="Extract content-creditability per from into talk logs")
    p.add_argument("--game-id", required=True)
    p.add_argument("--agent", action="append", help="Agent name (can be repeated). Default: all agents in game")
    p.add_argument("--base", type=Path, default=BASE_INFO_DIR)
    p.add_argument("--skip-pending", action="store_true", help="Skip pending entries")
    p.add_argument("--subdir-name", default=DEFAULT_OUT_SUBDIR, help="Output subdir (default: トーク履歴)")
    p.add_argument("--include-self", action="store_true", help="Include agent's own from entries")
    p.add_argument("--no-append", action="store_true", help="Do not append; overwrite from files")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        res = extract_pairs_for_game(
            base_dir=args.base,
            game_id=args.game_id,
            agents=args.agent,
            out_subdir=args.subdir_name,
            skip_pending=args.skip_pending,
            exclude_self=(not args.include_self),
            append=(not args.no_append),
        )
        total_agents = len(res)
        total_speakers = sum(len(v) for v in res.values())
        total_pairs = sum(sum(len(v2) for v2 in v.values()) for v in res.values())
        logger.info("Done. agents=%d speakers=%d pairs=%d", total_agents, total_speakers, total_pairs)
        return 0
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())