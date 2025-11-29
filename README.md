# FinNewsBot - Financial News Telegram Bot

A Telegram bot that delivers financial news about stocks and cryptocurrencies using Tavily web search.

## Features

- 🔍 **Web Search**: Integrates with Tavily API for real-time financial news search
- 🧠 **LLM Q&A**: `/query` command uses AI models via OpenRouter to answer questions with cited sources
- ⏰ **Customizable Intervals**: Set news update frequency and preferred delivery time
- 📈 **Financial Focus**: Specialized in stock market and cryptocurrency news with diversified categories (Bitcoin, other crypto, stocks, global markets)

## Prerequisites

- Python 3.12.9
- Telegram Bot Token (get it from [@BotFather](https://t.me/botfather))
- Tavily API Key (get it from [Tavily](https://tavily.com))
- OpenRouter API Key (get it from [OpenRouter](https://openrouter.ai))

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
OPENROUTER_API_KEY=your_openrouter_api_key_here
MODEL_NAME=x-ai/grok-2-1212  # optional override, defaults to Grok 2 1212 (free)
```

## Recommended Free Models

All these models are available for free on OpenRouter:

- **Grok 2 1212** (`x-ai/grok-2-1212`) - Default, fast and capable
- **DeepSeek R1** (`deepseek/deepseek-r1`) - Excellent reasoning capabilities  
- **Llama 3.1 70B** (`meta-llama/llama-3.1-70b-instruct`) - Strong general performance

You can change the model by setting `MODEL_NAME` in your `.env` file.

## Usage

1. Start the bot:
```bash
python main.py
```

2. Open Telegram and find your bot (search by the username you set with BotFather)

3. Send `/start` to begin

4. Get your news immediately with `/news`
or
Set your news interval using:
```
/set_interval 1 day
```

Optionally add a preferred delivery time (if not specified, updates start from current time):
```
/set_interval daily 7:00
/set_interval 1d 7AM
/set_interval weekly 9:30
```

Accepted time formats: `HH:MM` (24-hour) or `H[H]AM/PM`.

Available intervals:
- `3h` or `3 hours`
- `6h` or `6 hours`
- `12h` or `12 hours`
- `1d` or `1 day` or `daily`
- `3d` or `3 days`
- `1w` or `1 week` or `weekly`

## Commands

- `/start` - Start the bot and see welcome message
- `/set_interval <interval> [<time>]` - Set update frequency and optional delivery time (e.g., `/set_interval daily 7AM` or `/set_interval weekly 9:30`). If no time is specified, updates start from current time.
- `/news` - Receive the latest news update immediately on demand
- `/query <question>` - Ask any financial question answered by the LLM with Tavily-backed citations

## How It Works

1. The bot uses Tavily API to fetch the latest financial news about Bitcoin, other crypto, individual stocks, and global markets
2. Each update contains up to 5 items, one per category, so you don’t just see 5 Bitcoin headlines in a row
3. News updates are sent to users based on their configured intervals (and optional delivery time)
4. The bot checks every 5 minutes if any user is due for an update

## Configuration

### News Categories

`main.py` defines five Tavily search queries that map to the slots in each update:

1. Economy
2. Global stock/indices & macro news
3. Individual stocks
4. Crypto market
5. Foreign exchange
6. Precious metals

Adjust those query strings if you want different coverage.

### LLM Settings

The `/query` command uses OpenRouter API:

- `MODEL_NAME` defaults to `x-ai/grok-4.1-fast:free` (Grok 4.1 Fast)
- `OPENROUTER_API_KEY` is required for API access
- All recommended models (Grok 4.1 Fast, DeepSeek R1T2, Llama 3) are free on OpenRouter
- Responses cite the numbered Tavily sources (e.g., `[1]`)

## Data Storage

User preferences are stored in `user_data.json`. In production, consider using a proper database.

## Troubleshooting

- **API errors**: Verify your API keys (TELEGRAM_BOT_TOKEN, TAVILY_API_KEY, OPENROUTER_API_KEY) are correct in the `.env` file
- **No news updates**: Check that your interval is set correctly and the bot is running

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

