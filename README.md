# FinNewsBot - Financial News Telegram Bot

A Telegram bot that delivers financial news about stocks and cryptocurrencies using Tavily web search.

## Features

- 🔍 **Web Search**: Integrates with Tavily API for real-time financial news search
- 🧠 **LLM Q&A**: `/query` command uses DeepSeek-R1-Distill-Qwen-1.5B via Hugging Face Inference to answer questions with cited sources
- ⏰ **Customizable Intervals**: Set news update frequency and preferred delivery time
- 📈 **Financial Focus**: Specialized in stock market and cryptocurrency news with diversified categories (Bitcoin, other crypto, stocks, global markets)

## Prerequisites

- Python 3.12.9
- Telegram Bot Token (get it from [@BotFather](https://t.me/botfather))
- Tavily API Key (get it from [Tavily](https://tavily.com))
- Hugging Face API Token (needs access to `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`)

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
HF_MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  # optional override
```

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
/set_interval 1 day 7:00
/set_interval 1d 7AM
```

Accepted time formats: `HH:MM` (24-hour) or `H[H]AM/PM`.

Available intervals:
- `3h` or `3 hours`
- `6h` or `6 hours`
- `12h` or `12 hours`
- `1d` or `1 day`
- `3d` or `3 days`
- `1w` or `1 week`

## Commands

- `/start` - Start the bot and see welcome message
- `/set_interval <interval> [<time>]` - Set update frequency and optional delivery time (e.g., `/set_interval 1 day 7AM`). If no time is specified, updates start from current time.
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

The `/query` command uses the Hugging Face Inference API:

- `HF_MODEL_NAME` defaults to `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `HF_TOKEN` must have access to this model
- Responses cite the numbered Tavily sources (e.g., `[1]`)

## Data Storage

User preferences are stored in `user_data.json`. In production, consider using a proper database.

## Troubleshooting

- **API errors**: Verify your API keys (TELEGRAM_BOT_TOKEN, TAVILY_API_KEY) are correct in the `.env` file
- **No news updates**: Check that your interval is set correctly and the bot is running

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

