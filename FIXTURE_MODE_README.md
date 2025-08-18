# Analysis Fixture Mode

## 概要

Fixture モードは、サーバー側の不具合で `talk_history` が `Skip`/`Over` になっても、クライアント側だけでE2E（履歴 → 分析 → ファイル出力）を実運用中に検証できる機能です。

## 主な特徴

- **置換は分析専用**: ローカルコピー（代理Talk）で行うため、サーバー送信内容や元履歴は変更されません
- **スイッチ制御**: ON/OFF可能（デフォルトOFF）
- **ファイル分離**: 本番の `analysis.yml` とは別の `analysis_test.yml` に保存
- **トレース記録**: 置換の痕跡（元 → 置換）を `analysis_fixture_trace.yml` に記録
- **下流処理制御**: Fixtureモード中は `select_sentence.yml`/`intention.yml` の更新を既定で停止

## 設定方法

### 1. config.ymlでの設定

```yaml
analysis:
  fixture_mode:
    enable: true                    # Fixtureモードを有効化
    rewrite_targets: ["Skip","Over"] # 置換対象テキスト
    output_file: "analysis_test.yml" # 出力ファイル名
    trace_file: "analysis_fixture_trace.yml"  # トレースファイル
    max_per_call: 5                  # 1回の呼び出しで処理する最大件数
    apply_to_agents: "others"        # "others"（他者のみ）| "all"（自分含む）
    utterances:
      default:
        - "占いCOの有無を確認したいです。"
        - "便乗と早い同調を重く見ます。"
      by_agent:                      # エージェント別の置換テキスト（任意）
        メイ:
          - "メイさん、方針を教えてください。"
```

### 2. 環境変数での設定（config.ymlより優先）

```bash
# Fixtureモードの有効化
export ANALYSIS_FIXTURE_ENABLE=1

# 出力ファイル名の変更
export ANALYSIS_FIXTURE_OUTPUT_FILE=my_test.yml

# トレースファイル名の変更
export ANALYSIS_FIXTURE_TRACE_FILE=my_trace.yml

# 置換対象テキスト（カンマ区切り）
export ANALYSIS_FIXTURE_TARGETS="Skip,Over,Null"

# 最大処理件数
export ANALYSIS_FIXTURE_MAX_PER_CALL=10

# 適用範囲（others または all）
export ANALYSIS_FIXTURE_APPLY_TO=all

# デフォルト置換テキスト（|区切り）
export ANALYSIS_FIXTURE_UTTERANCES_DEFAULT="テキスト1|テキスト2|テキスト3"

# 下流処理の強制実行（Fixtureモードでも実行）
export ANALYSIS_UPDATE_SELECT_SENTENCE=1
export ANALYSIS_UPDATE_INTENTION=1
```

## 動作仕様

### 置換条件

以下の条件をすべて満たす場合に置換が実行されます：

1. Fixtureモードが有効（`enable: true`）
2. テキストが `rewrite_targets` に含まれる、または `skip=True`/`over=True` フラグ
3. `apply_to_agents` が `"all"` または発話者が自分以外

### 置換処理

1. 元のTalkオブジェクトは変更せず、`SimpleNamespace`による代理Talkを作成
2. 代理Talkには `text=置換テキスト`, `skip=False`, `over=False` を設定
3. 置換テキストは設定された候補からローテーション選択（安定性重視）
4. 1回の呼び出しで `max_per_call` 件まで処理

### ファイル出力

- **通常モード**: `analysis.yml` に保存、下流処理も実行
- **Fixtureモード**: `analysis_test.yml` に保存、下流処理は既定でスキップ

### トレース記録

`analysis_fixture_trace.yml` に以下の形式で記録：

```yaml
1:1:3da47453:
  from_agent: Agent1
  original: Skip
  replaced: 占いCOの有無を確認したいです。
  timestamp: '2025-08-18T13:14:15.904223'
```

## 使用例

### テスト実行

```python
# test_fixture.py
from utils.bdi.micro_bdi.analysis_tracker import AnalysisTracker

config = {
    "analysis": {
        "fixture_mode": {
            "enable": True,
            "output_file": "analysis_test.yml"
        }
    }
}

tracker = AnalysisTracker(config, "TestAgent", "GAME_ID")

# Skip/Overを含むトーク履歴
talks = [
    Talk(agent="Agent1", text="Skip"),
    Talk(agent="Agent2", text="Over"),
]

# 分析実行（Skip/Overが置換されて分析される）
added = tracker.analyze_talk(talks, info)
if added > 0:
    tracker.save_analysis()  # analysis_test.ymlに保存
```

### 実行ログ例

```
[AnalysisTracker] Fixture mode ENABLED: output=analysis_test.yml, max=5
[AnalysisTracker] Fixture replacement: 'Skip' -> '占いCOの有無を確認したいです。' for Agent1
[AnalysisTracker] Fixture replacement: 'Over' -> '便乗と早い同調を重く見ます。' for Agent2
[AnalysisTracker] saved: analysis_test.yml size=368
[AnalysisTracker] Downstream(select_sentence): SKIP (disabled by fixture)
[AnalysisTracker] Downstream(intention): SKIP (disabled by fixture)
```

## 受け入れ基準

1. **デフォルト互換性**: Fixture無効時は従来と完全互換
2. **置換動作**: Skip/Overが置換テキストで分析される
3. **ファイル分離**: `analysis_test.yml` に保存される
4. **トレース記録**: 置換の対応が記録される
5. **元データ不変**: 元ログ・送信文字列は変更されない
6. **下流制御**: 既定でスキップ、環境変数で上書き可能
7. **上限制御**: `max_per_call` を超える件数でも制限される

## トラブルシューティング

### Fixtureモードが動作しない

1. `config.yml` の `analysis.fixture_mode.enable` が `true` か確認
2. 環境変数 `ANALYSIS_FIXTURE_ENABLE` が設定されていないか確認

### 置換されない

1. `rewrite_targets` に対象テキストが含まれているか確認
2. `apply_to_agents` の設定が適切か確認（自分の発話は `"all"` が必要）

### ファイルが生成されない

1. 書き込み権限があるか確認
2. `output_file` のパスが正しいか確認
3. 分析対象のトークが存在するか確認