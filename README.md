# Arcane Registry

A tool to catalog Magic: The Gathering cards. Upload photos of your cards, and the app uses AI vision to identify them, validates against Scryfall, and exports to CSV.

## Setup
0. Clone the repo

1. Copy `env.example` to `.env` and add your API key:

```
VLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

2. Run:

```
cd path/to/repo/ && docker compose up --build
```

3. Open http://localhost:5050

## Supported Providers

| Provider | Variable |
|----------|----------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |

Set `VLM_PROVIDER` to `openai`, `anthropic`, or `deepseek`.

## License

For personal use. Card data is property of Wizards of the Coast.
