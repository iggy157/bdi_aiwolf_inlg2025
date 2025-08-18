# Fixture Mode - Complete Documentation

## Overview

Fixture Mode allows the analysis system to work even when the server only returns Skip/Over messages. Instead of analyzing empty or meaningless utterances, it replaces them with predefined meaningful text for analysis purposes, while keeping all original data unchanged.

## Key Features

- **Environment-only activation**: Can be enabled with just `ANALYSIS_FIXTURE_ENABLE=1`, no config.yml needed
- **Safe replacement**: Original talk history is never modified, only analysis uses substitute text
- **Separate output**: Saves to `analysis_test.yml` to avoid contaminating production `analysis.yml`
- **Trace recording**: Logs all replacements in `analysis_fixture_trace.yml` for transparency
- **Carryover support**: Handles `max_per_call` limits properly, processing remaining items in subsequent calls
- **Default prompts**: Works without config.yml prompts using built-in English defaults

## Environment Variables

### Core Settings
```bash
# Enable Fixture mode (required)
export ANALYSIS_FIXTURE_ENABLE=1

# Set processing limit per call (default: 999)
export ANALYSIS_FIXTURE_MAX_PER_CALL=999

# Define target texts to replace (default: "Skip,Over")
export ANALYSIS_FIXTURE_TARGETS="Skip,Over"

# Output file name (default: "analysis_test.yml")
export ANALYSIS_FIXTURE_OUTPUT_FILE="analysis_test.yml"

# Trace file name (default: "analysis_fixture_trace.yml")
export ANALYSIS_FIXTURE_TRACE_FILE="analysis_fixture_trace.yml"

# Apply to which agents: "others" or "all" (default: "others")
export ANALYSIS_FIXTURE_APPLY_TO="others"

# Default replacement texts (| separated)
export ANALYSIS_FIXTURE_UTTERANCES_DEFAULT="Are there any seer claims?|Please give a tentative vote with a reason.|Share one town and one wolf read."
```

### Downstream Control
```bash
# Force enable downstream processing (normally disabled in Fixture mode)
export ANALYSIS_UPDATE_SELECT_SENTENCE=1
export ANALYSIS_UPDATE_INTENTION=1
```

## Usage Examples

### Minimal Setup (Environment only)
```bash
export ANALYSIS_FIXTURE_ENABLE=1
export ANALYSIS_FIXTURE_UTTERANCES_DEFAULT="Are there any seer claims?|Please give a tentative vote with a reason.|Share one town and one wolf read."
python src/main.py
```

### Production Testing Setup
```bash
export ANALYSIS_FIXTURE_ENABLE=1
export ANALYSIS_FIXTURE_MAX_PER_CALL=999
export ANALYSIS_FIXTURE_OUTPUT_FILE="production_test.yml"
export ANALYSIS_FIXTURE_UTTERANCES_DEFAULT="占いCOはありますか？|仮投票先と理由をお願いします。|怪しいと思う人を教えてください。"
python src/main.py
```

### Incremental Processing (Testing carryover)
```bash
export ANALYSIS_FIXTURE_ENABLE=1
export ANALYSIS_FIXTURE_MAX_PER_CALL=1
python src/main.py
```

## Log Indicators

### Fixture Mode Activation
```
[AnalysisTracker] Fixture mode ENABLED: output=analysis_test.yml, max=999, apply_to=others
```

### Replacement Occurrence
```
[AnalysisTracker] Fixture replacement: 'Skip' -> 'Are there any seer claims?' for Agent1
[AnalysisTracker] Fixture replacement: 'Over' -> 'Please give a tentative vote with a reason.' for Agent2
```

### Processing Status
```
[AnalysisTracker] Found 5 candidates, processing 5
[AnalysisTracker] 2 candidates deferred to next call
```

### File Output
```
[AnalysisTracker] saved: /path/to/analysis_test.yml size=588
[AnalysisTracker] Downstream(select_sentence): SKIP (disabled by fixture)
[AnalysisTracker] Downstream(intention): SKIP (disabled by fixture)
```

### Default Prompt Usage
```
[AnalysisTracker] Using default prompt for analyze_message_type
[AnalysisTracker] Using default prompt for analyze_target_agents
[AnalysisTracker] Using default prompt for analyze_credibility
```

## Output Files

### Analysis Output: `analysis_test.yml`
```yaml
1:
  content: Are there any seer claims?
  type: question
  from: Agent1
  to: 'null'
  credibility: 0.7125
2:
  content: Please give a tentative vote with a reason.
  type: co
  from: Agent2
  to: 'null'
  credibility: 0.775
```

### Trace File: `analysis_fixture_trace.yml`
```yaml
1:1:3da47453:
  from_agent: Agent1
  original: Skip
  replaced: Are there any seer claims?
  timestamp: '2025-08-18T14:14:45.048326'
1:2:18a63f55:
  from_agent: Agent2
  original: Over
  replaced: Please give a tentative vote with a reason.
  timestamp: '2025-08-18T14:15:01.246570'
```

## File Locations

All files are saved under:
```
info/bdi_info/micro_bdi/<game_id>/<agent_name>/
├── analysis_test.yml          # Fixture analysis results
├── analysis_fixture_trace.yml # Replacement history
└── analysis.yml               # Normal analysis (when Fixture OFF)
```

## Default Replacement Texts

When no custom texts are provided, these English defaults are used:
1. "Are there any seer claims?"
2. "Please give a tentative vote with a reason."
3. "Share one town and one wolf read."
4. "Who do you think is suspicious and why?"
5. "What is your current analysis of the situation?"

## Default Analysis Prompts

When config.yml lacks prompt definitions, built-in English prompts are used:

### Message Type Classification
- Returns: `co`, `question`, `positive`, `negative`, or `null`

### Target Agent Detection  
- Returns: comma-separated agent names, `all`, or `null`

### Credibility Scoring
- Returns four 0-1 metrics: `logical_consistency`, `specificity_and_detail`, `intuitive_depth`, `clarity_and_conciseness`

## Safety Features

### Data Integrity
- Original talk history is never modified
- Server communication remains unchanged
- Production `analysis.yml` is untouched when Fixture is active

### Carryover Protection
- Unprocessed candidates are preserved for next call
- No analysis is lost due to `max_per_call` limits
- Seen-talk tracking prevents duplicate processing

### Fallback Robustness
- Works without any config.yml configuration
- Built-in default prompts and replacement texts
- Graceful handling of missing LLM or prompt errors

## Troubleshooting

### Fixture Mode Not Activating
1. Verify `ANALYSIS_FIXTURE_ENABLE=1` is set
2. Check logs for "Fixture mode ENABLED" message
3. Ensure environment variables are exported correctly

### No Replacements Occurring
1. Check `ANALYSIS_FIXTURE_TARGETS` matches actual talk text
2. Verify `ANALYSIS_FIXTURE_APPLY_TO` setting (others vs all)
3. Look for "Fixture replacement" log messages

### Files Not Generated
1. Confirm `added > 0` in logs (indicates successful analysis)
2. Check write permissions in output directory
3. Verify `ANALYSIS_FIXTURE_OUTPUT_FILE` path is valid

### Carryover Not Working
1. Set `ANALYSIS_FIXTURE_MAX_PER_CALL` to small value (e.g., 1)
2. Look for "candidates deferred to next call" log message
3. Verify subsequent calls process remaining items

## Production Considerations

### Performance
- Set reasonable `ANALYSIS_FIXTURE_MAX_PER_CALL` limits
- Monitor LLM API usage when processing many replacements
- Consider batch processing for large talk histories

### Quality Assurance
- Review trace files to verify appropriate replacements
- Monitor analysis results for meaningful classifications
- Compare Fixture vs normal analysis outputs for consistency

### Operational Safety
- Always use separate output files in production
- Keep downstream processing disabled unless specifically needed
- Maintain clear logging to distinguish Fixture from normal operation