# FinNewsBot - Financial News Telegram Bot

A Telegram bot that delivers financial news about stocks and cryptocurrencies using Tavily web search with timezone-aware scheduling.

## Features

- **Web Search**: Integrates with Tavily API for real-time financial news search
- **LLM Q&A**: `/query` command uses AI models via OpenRouter to answer questions with cited sources
- **Timezone Support**: Set your timezone for accurate delivery times
- **Customizable Intervals**: Set news update frequency and preferred delivery time
- **Financial Focus**: Specialized in stock market and cryptocurrency news with diversified categories
- **SQLite Storage**: Reliable database storage for user preferences

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
```

## Usage

1. Start the bot:
```bash
python main.py
```

2. Open Telegram and find your bot (search by the username you set with BotFather)

3. Send `/start` to begin

4. Set your news schedule with timezone:
```
/set_interval daily 9:00 UTC+7
/set_interval weekly 15:30 Asia/Bangkok
/set_interval 12h EST
```

## Commands

- `/start` - Start the bot and see welcome message
- `/set_interval <frequency> [time] [timezone]` - Set update frequency, time, and timezone
- `/news` - Get immediate news update
- `/status` - Check your current settings and next update time
- `/query <question>` - Ask financial questions with AI-powered answers
- `/timezone [timezone]` - View or set your timezone

## Interval Options

- **Frequency**: `3h`, `6h`, `12h`, `daily`, `3d`, `weekly`
- **Time**: `9:00`, `15:30`, `7AM`, `3PM` (optional)
- **Timezone**: `UTC+7`, `Asia/Bangkok`, `EST`, `PST`, etc. (optional)

## Examples

### Basic Setup
```
/set_interval daily
/set_interval 12h
/set_interval weekly
```

### With Specific Time
```
/set_interval daily 9:00
/set_interval weekly 15:30
/set_interval 12h 8:00
```

### With Timezone
```
/set_interval daily 9:00 UTC+7
/set_interval weekly 15:30 Asia/Bangkok
/set_interval 12h EST
```

## Timezone Support

### Supported Formats
- **UTC Offsets**: `+7`, `-5`, `UTC+8`, `GMT-3`
- **Timezone Names**: `Asia/Bangkok`, `Europe/London`, `America/New_York`
- **Common Codes**: `EST`, `PST`, `CET`, `JST`, `IST`

### How It Works
- All times are stored in UTC internally
- Display times are converted to your timezone
- Scheduled deliveries respect your local time
- Default timezone is UTC if not set

## News Categories

The bot fetches news from 6 categories:
1. **Economy** - Economic overview and indicators
2. **Global Stock Market** - Market indices and macro news
3. **Individual Stocks** - Company-specific news
4. **Crypto Market** - Cryptocurrency market updates
5. **Forex Market** - Foreign exchange news
6. **Precious Metals** - Gold, silver, and metals market

## Data Storage

- **Database**: SQLite (`user_data.db`)
- **Migration**: Automatic migration from JSON if `user_data.json` exists
- **Backup**: Use `migrate_to_sqlite.py` for manual migration

## Configuration

### LLM Settings
- **Model**: `meta-llama/llama-3.3-70b-instruct:free` (via OpenRouter)
- **Max Tokens**: 400
- **Temperature**: 0.6

### Update Schedule
- **Check Interval**: Every 5 minutes
- **Timezone Aware**: All calculations respect user timezones
- **Error Handling**: Comprehensive error handling with user feedback

## Troubleshooting

### Common Issues
- **API errors**: Verify API keys in `.env` file
- **No updates**: Check interval settings with `/status`
- **Wrong time**: Set correct timezone with `/timezone`
- **Bot conflict**: Only run one instance at a time

### Error Messages
- **Invalid timezone**: Use supported timezone formats
- **Invalid interval**: Use valid frequency options
- **Database errors**: Check file permissions for SQLite

## Migration from JSON

If upgrading from a previous version:
```bash
python migrate_to_sqlite.py
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.