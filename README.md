# aiwolf-nlp-agent-llm

[README in English](/README.en.md)

人狼知能コンテスト（自然言語部門） のLLMを用いたサンプルエージェントです。

## 環境構築

> [!IMPORTANT]
> Python 3.11以上が必要です。

```bash
git clone https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git
cd aiwolf-nlp-agent-llm
cp config/.env.example config/.env
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 日本語のプロンプトを使用したい場合
```bash
cp config/config.jp.yml.example config/config.yml
```

### 英語のプロンプトを使用したい場合
```bash
cp config/config.en.yml.example config/config.yml
```

## その他

実行方法や設定などその他については[aiwolf-nlp-agent](https://github.com/aiwolfdial/aiwolf-nlp-agent)をご確認ください。

## サーバー起動
### 🐧 Linux

```bash
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/aiwolf-nlp-server-linux-amd64
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_5.yml
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_13.yml
curl -Lo .env https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/example.env
chmod u+x ./aiwolf-nlp-server-linux-amd64
./aiwolf-nlp-server-linux-amd64 -c ./default_5.yml # 5人ゲーム
# ./aiwolf-nlp-server-linux-amd64 -c ./default_13.yml # 13人ゲーム
```

## ビュアー起動

```bash
cd aiwolf-nlp-viewer/
pnpm run dev --open
```