# FinNewsBot - Financial News Telegram Bot

A Telegram bot that delivers financial news about stocks and cryptocurrencies using Hugging Face LLM models and Tavily web search.

## Features

- 🤖 **AI-Powered**: Uses Hugging Face LLM models to summarize financial news
- 🔍 **Web Search**: Integrates with Tavily API for real-time financial news search
- ⏰ **Customizable Intervals**: Set news update frequency from 3 hours to 1 week
- 📈 **Financial Focus**: Specialized in stock market and cryptocurrency news

## Prerequisites

- Python 3.12.9
- Telegram Bot Token (get it from [@BotFather](https://t.me/botfather))
- Tavily API Key (get it from [Tavily](https://tavily.com))
- Hugging Face API Token (get it from [Hugging Face](https://huggingface.co/settings/tokens))

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd FinNewsBot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file from the example:
```bash
cp .env.example .env
```

4. Edit `.env` and add your credentials:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TAVILY_API_KEY=your_tavily_api_key_here
HF_TOKEN=your_huggingface_api_token_here
```

## Usage

1. Start the bot:
```bash
python main.py
```

2. Open Telegram and find your bot (search by the username you set with BotFather)

3. Send `/start` to begin

4. Set your news interval using:
```
/set_interval 1 day
```

Available intervals:
- `3h` or `3 hours`
- `6h` or `6 hours`
- `12h` or `12 hours`
- `1d` or `1 day`
- `3d` or `3 days`
- `1w` or `1 week`

## Commands

- `/start` - Start the bot and see welcome message
- `/set_interval <interval>` - Set how frequently you want to receive news updates
- `/news` - Receive the latest news update immediately on demand

## How It Works

1. The bot uses Tavily API to search for the latest financial news about stocks and cryptocurrencies
2. The news is processed through a Hugging Face LLM model to generate summaries
3. News updates are sent to users based on their configured intervals
4. The bot checks every 5 minutes if any user is due for an update

## Configuration

### Hugging Face Models

You can change the LLM model by setting `HF_MODEL_NAME` in your `main.py` file. The bot uses Hugging Face's Inference API, so you don't need to download models locally. Some recommended models:
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (default) - Lightweight, fast, great for quick inference
- `meta-llama/Llama-3.1-8B-Instruct` - High-quality instruction following
- `mistralai/Mistral-7B-Instruct-v0.2` - Fast and efficient
- `google/gemma-7b-it` - Google's Gemma model

Note: The bot uses Hugging Face's Inference API, so no local model download is required. Make sure your HF_TOKEN has access to the model you choose.

## Data Storage

User preferences are stored in `user_data.json`. In production, consider using a proper database.

## Troubleshooting

- **API errors**: Verify your API keys (TELEGRAM_BOT_TOKEN, TAVILY_API_KEY, HF_TOKEN) are correct in the `.env` file
- **No news updates**: Check that your interval is set correctly and the bot is running
- **HF_TOKEN issues**: Make sure your Hugging Face token has access to the model you're using

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

