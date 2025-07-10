# aiwolf-nlp-agent-llm

[README in Japanese](/README.md)

This is a sample agent using LLM for the AIWolf Competition (Natural Language Division).

## Environment Setup

> [!IMPORTANT]
> Python 3.11 or higher is required.

```bash
git clone https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git
cd aiwolf-nlp-agent-llm
cp config/config.yml.example config/config.yml
cp config/.env.example config/.env
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Others

For details on execution methods, settings, and other information, please refer to [aiwolf-nlp-agent](https://github.com/aiwolfdial/aiwolf-nlp-agent).
