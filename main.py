import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import httpx

# Load environment variables
load_dotenv()

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_API_URL = "https://api.tavily.com/search"

# User data storage (in production, use a database)
USER_DATA_FILE = "user_data.json"

def load_user_data() -> Dict:
    """Load user data from JSON file."""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_data(data: Dict):
    """Save user data to JSON file."""
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def get_user_interval(user_id: int) -> int:
    """Get user's news interval in seconds."""
    data = load_user_data()
    return data.get(str(user_id), {}).get("interval_seconds", None)

def set_user_interval(user_id: int, interval_seconds: int, send_time: Optional[str] = None):
    """Set user's news interval and optional preferred send time."""
    data = load_user_data()
    if str(user_id) not in data:
        data[str(user_id)] = {}
    data[str(user_id)]["interval_seconds"] = interval_seconds
    data[str(user_id)]["last_sent"] = None
    data[str(user_id)]["send_time"] = send_time
    save_user_data(data)

def parse_time_input(time_input: str) -> Optional[str]:
    """Parse user provided time strings such as 7AM, 7:00, or 19:30."""
    if not time_input:
        return None
    cleaned = time_input.strip().lower()
    if not cleaned:
        return None
    cleaned = cleaned.replace(".", "")
    am_pm = None
    if cleaned.endswith("am") or cleaned.endswith("pm"):
        am_pm = cleaned[-2:]
        cleaned = cleaned[:-2]
    cleaned = cleaned.strip()
    if ":" in cleaned:
        hour_part, minute_part = cleaned.split(":", 1)
    else:
        hour_part, minute_part = cleaned, "0"
    if not hour_part.isdigit() or not minute_part.isdigit():
        return None
    hour = int(hour_part)
    minute = int(minute_part)
    if am_pm:
        if hour < 1 or hour > 12:
            return None
        if am_pm == "am":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"

def extract_interval_and_time(raw_input: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Split the interval string and optional time component.
    Returns tuple of (interval_string, normalized_time, error_message_if_any)
    """
    interval_part = raw_input.strip()
    if not interval_part:
        return "", None, None
    # If explicit separator
    if "-" in interval_part:
        left, right = interval_part.split("-", 1)
        interval = left.strip()
        time_candidate = right.strip()
        normalized = parse_time_input(time_candidate) if time_candidate else None
        if time_candidate and not normalized:
            return interval, None, "Invalid time format. Use values like 7AM or 07:00."
        return interval, normalized, None
    tokens = interval_part.split()
    normalized = None
    interval_tokens = tokens[:]
    # Try last two tokens first (e.g., "7 PM")
    for take in (2, 1):
        if len(tokens) > take:
            candidate = " ".join(tokens[-take:])
            parsed = parse_time_input(candidate)
            if parsed:
                interval_tokens = tokens[:-take]
                normalized = parsed
                break
    interval = " ".join(interval_tokens).strip()
    return interval, normalized, None

def align_to_send_time(reference: datetime, send_time_str: str) -> datetime:
    """Align the reference datetime to the next occurrence of the preferred send time."""
    hour, minute = map(int, send_time_str.split(":"))
    candidate = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate < reference:
        candidate += timedelta(days=1)
    return candidate

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command - greet user and ask about frequency."""
    user_id = update.effective_user.id
    username = update.effective_user.first_name
    
    welcome_message = (
        f"Hello {username}! 👋\n\n"
        f"Welcome to FinNewsBot! 📈💰\n\n"
        f"I'm here to keep you updated with the latest financial news about stocks and cryptocurrencies.\n\n"
        f"To get started, please set how frequently you'd like to receive news updates using the command:\n"
        f"/set_interval\n\n"
        f"You can choose from:\n"
        f"• 3 hours\n"
        f"• 6 hours\n"
        f"• 12 hours\n"
        f"• 1 day\n"
        f"• 3 days\n"
        f"• 1 week\n\n"
        f"Example: /set_interval 1 day"
    )
    
    await update.message.reply_text(welcome_message)

async def set_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /set_interval command."""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "Please specify an interval. Available options:\n"
            "• 3h or 3 hours\n"
            "• 6h or 6 hours\n"
            "• 12h or 12 hours\n"
            "• 1d or 1 day\n"
            "• 3d or 3 days\n"
            "• 1w or 1 week\n\n"
            "Example: /set_interval 1 day"
        )
        return
    
    # Parse interval and optional send time
    raw_interval_input = " ".join(context.args)
    interval_str, send_time, error = extract_interval_and_time(raw_interval_input)
    if error:
        await update.message.reply_text(error)
        return
    interval_seconds = None
    
    interval_map = {
        "3h": 3 * 3600, "3 hours": 3 * 3600, "3hour": 3 * 3600,
        "6h": 6 * 3600, "6 hours": 6 * 3600, "6hour": 6 * 3600,
        "12h": 12 * 3600, "12 hours": 12 * 3600, "12hour": 12 * 3600,
        "1d": 24 * 3600, "1 day": 24 * 3600, "1day": 24 * 3600,
        "3d": 3 * 24 * 3600, "3 days": 3 * 24 * 3600, "3day": 3 * 24 * 3600,
        "1w": 7 * 24 * 3600, "1 week": 7 * 24 * 3600, "1week": 7 * 24 * 3600,
    }
    
    interval_seconds = interval_map.get(interval_str.lower())
    
    if interval_seconds is None:
        await update.message.reply_text(
            "Invalid interval. Please use one of:\n"
            "• 3h or 3 hours\n"
            "• 6h or 6 hours\n"
            "• 12h or 12 hours\n"
            "• 1d or 1 day\n"
            "• 3d or 3 days\n"
            "• 1w or 1 week"
        )
        return
    
    set_user_interval(user_id, interval_seconds, send_time)
    
    # Format interval for display
    if interval_seconds < 3600:
        display = f"{interval_seconds // 60} minutes"
    elif interval_seconds < 86400:
        display = f"{interval_seconds // 3600} hours"
    elif interval_seconds < 604800:
        display = f"{interval_seconds // 86400} days"
    else:
        display = f"{interval_seconds // 604800} week(s)"
    
    time_notice = f" at {send_time}" if send_time else ""
    await update.message.reply_text(
        f"✅ Interval set to {display}{time_notice}!\n\n"
        f"You will receive financial news updates every {display}{time_notice}.\n"
        f"Your first update will be sent shortly."
    )
    
    # Send first news update immediately
    await send_financial_news(update.message.chat_id, context)

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /news command - send immediate news update."""
    chat_id = update.effective_chat.id
    
    # Optional acknowledgement to show the bot is working
    await update.message.reply_text("Fetching the latest financial news for you... ⏳")
    
    await send_financial_news(chat_id, context)

async def search_financial_news(query: str, max_results: int = 1) -> List[Dict]:
    """Search for financial news using Tavily API and return raw results list."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                TAVILY_API_URL,
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": max_results,
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            search_results = response.json()
        return search_results.get("results", [])[:max_results] if search_results else []
    except Exception as e:
        return [{"title": "Error searching for news", "url": "", "content": str(e)}]

async def send_financial_news(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Send financial news to user."""
    try:
        # Prepare diverse news categories
        categories = [
            ("Bitcoin", "latest bitcoin news today"),
            ("Other Crypto", "latest cryptocurrency news excluding bitcoin today"),
            ("Crypto Market", "latest cryptocurrency market overview today"),
            ("Individual Stocks", "latest individual company stock news today"),
            ("Global Stock Market", "latest global stock market indices and macroeconomic news today"),
        ]

        sections: List[str] = []
        for label, query in categories:
            results = await search_financial_news(query, max_results=1)
            if results:
                item = results[0]
                title = item.get("title", "No title")
                url = item.get("url", "")
                content = item.get("content", "")[:200]
                sections.append(
                    f"**{label}**\n"
                    f"{title}\n"
                    f"{content}...\n"
                    f"🔗 {url}\n"
                )
            else:
                sections.append(f"**{label}**\nNo news found for this category right now.\n")
        
        # Format message
        numbered_sections = []
        for idx, sec in enumerate(sections, start=1):
            numbered_sections.append(f"{idx}. {sec}")

        message = (
            "📈 **Financial News Update** 📈\n\n"
            + "\n".join(numbered_sections)
        )
        
        # Send message (split if too long)
        if len(message) > 4096:
            # Split message into chunks
            chunks = [message[i:i+4096] for i in range(0, len(message), 4096)]
            for chunk in chunks:
                await context.bot.send_message(chat_id=chat_id, text=chunk, parse_mode="Markdown")
        else:
            await context.bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
        
        # Update last sent time
        data = load_user_data()
        if str(chat_id) in data:
            data[str(chat_id)]["last_sent"] = datetime.now().isoformat()
            save_user_data(data)
    
    except Exception as e:
        error_msg = f"Error sending news: {str(e)}"
        try:
            await context.bot.send_message(chat_id=chat_id, text=error_msg)
        except:
            print(f"Failed to send error message: {error_msg}")

async def check_and_send_updates(context: ContextTypes.DEFAULT_TYPE):
    """Check all users and send updates if interval has passed."""
    data = load_user_data()
    current_time = datetime.now()
    
    for user_id_str, user_data in data.items():
        interval_seconds = user_data.get("interval_seconds")
        if not interval_seconds:
            continue
        
        last_sent_str = user_data.get("last_sent")
        send_time_str = user_data.get("send_time")
        
        if last_sent_str:
            last_sent = datetime.fromisoformat(last_sent_str)
            base_time = last_sent + timedelta(seconds=interval_seconds)
        else:
            base_time = current_time
        
        due_time = base_time
        if send_time_str:
            due_time = align_to_send_time(base_time, send_time_str)
        
        if current_time < due_time:
            continue
        
        # Send update
        try:
            await send_financial_news(int(user_id_str), context)
        except Exception as e:
            print(f"Error sending update to user {user_id_str}: {e}")

def main():
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found in environment variables")
    
    print("Bot configuration loaded successfully!")
    print(f"Using Hugging Face model: {HF_MODEL_NAME}")
    print(f"Using endpoint: {HF_API_URL}")
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("set_interval", set_interval))
    application.add_handler(CommandHandler("news", news))
    
    # Start periodic task
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(
            check_and_send_updates,
            interval=300,  # Check every 5 minutes
            first=10  # Start after 10 seconds
        )
    
    print("Bot is starting...")
    # Run bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

