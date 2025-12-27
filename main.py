import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import httpx
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"
TAVILY_API_URL = "https://api.tavily.com/search"

# User data storage (SQLite database)
DATABASE_FILE = "user_data.db"

def init_database():
    """Initialize SQLite database with user_settings table."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            interval_seconds INTEGER,
            last_sent TEXT,
            send_time TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_user_interval(user_id: int) -> int:
    """Get user's news interval in seconds."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT interval_seconds FROM user_settings WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def set_user_interval(user_id: int, interval_seconds: int, send_time: Optional[str] = None):
    """Set user's news interval and optional preferred send time."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Set last_sent to a time in the past to trigger immediate first news
    past_time = datetime.now() - timedelta(seconds=interval_seconds + 1)
    last_sent = past_time.isoformat()
    
    cursor.execute('''
        INSERT OR REPLACE INTO user_settings (user_id, interval_seconds, last_sent, send_time)
        VALUES (?, ?, ?, ?)
    ''', (user_id, interval_seconds, last_sent, send_time))
    
    conn.commit()
    conn.close()

def get_user_data(user_id: int) -> Dict:
    """Get all user data."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT interval_seconds, last_sent, send_time FROM user_settings WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "interval_seconds": result[0],
            "last_sent": result[1],
            "send_time": result[2]
        }
    return {}

def update_last_sent(user_id: int, last_sent: str):
    """Update the last_sent timestamp for a user."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('UPDATE user_settings SET last_sent = ? WHERE user_id = ?', (last_sent, user_id))
    conn.commit()
    conn.close()

def get_all_users() -> List[Tuple[int, Dict]]:
    """Get all users and their settings."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, interval_seconds, last_sent, send_time FROM user_settings')
    results = cursor.fetchall()
    conn.close()
    
    users = []
    for row in results:
        user_data = {
            "interval_seconds": row[1],
            "last_sent": row[2],
            "send_time": row[3]
        }
        users.append((row[0], user_data))
    
    return users

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
        f"Hello {username}! \n\n"
        f"Welcome to FinNewsBot! \n\n"
        f"I'm here to keep you updated with the latest financial news about stocks and cryptocurrencies.\n\n"
        f"To get started, please set how frequently you'd like to receive news updates using the command:\n"
        f"/set_interval\n\n"
        f"You can choose from:\n"
        f"• 3 hours\n"
        f"• 6 hours\n"
        f"• 12 hours\n"
        f"• 1 day / daily\n"
        f"• 3 days\n"
        f"• 1 week / weekly\n\n"
        f"You can optionally specify a preferred time (if not specified, updates start from current time):\n"
        f"Examples:\n"
        f"• /set_interval daily (starts from current time)\n"
        f"• /set_interval 1d 7AM (daily at 7 AM)\n"
        f"• /set_interval weekly 9:30 (weekly at 9:30)\n"
        f"• /set_interval 12h 9:30 (every 12 hours at 9:30)\n\n"
        f"Other commands:\n"
        f"• /news - Get immediate news update\n"
        f"• /status - Check your current settings\n"
        f"• /query <question> - Ask a financial question"
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
            "• 1d or 1 day or daily\n"
            "• 3d or 3 days\n"
            "• 1w or 1 week or weekly\n\n"
            "You can optionally specify a preferred time (if not specified, updates start from current time):\n"
            "• /set_interval 1 day 7AM\n"
            "• /set_interval 1d 7:00\n"
            "• /set_interval 12 hours 9:30\n\n"
            "Examples:\n"
            "• /set_interval daily (starts from current time)\n"
            "• /set_interval 1d 7AM (daily at 7 AM)\n"
            "• /set_interval weekly 9:30 (weekly at 9:30)\n"
            "• /set_interval 12h 9:30 (every 12 hours at 9:30)"
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
        "1d": 24 * 3600, "1 day": 24 * 3600, "1day": 24 * 3600, "daily": 24 * 3600,
        "3d": 3 * 24 * 3600, "3 days": 3 * 24 * 3600, "3day": 3 * 24 * 3600,
        "1w": 7 * 24 * 3600, "1 week": 7 * 24 * 3600, "1week": 7 * 24 * 3600, "weekly": 7 * 24 * 3600,
    }
    
    interval_seconds = interval_map.get(interval_str.lower())
    
    if interval_seconds is None:
        await update.message.reply_text(
            "Invalid interval. Please use one of:\n"
            "• 3h or 3 hours\n"
            "• 6h or 6 hours\n"
            "• 12h or 12 hours\n"
            "• daily or 1d or 1 day\n"
            "• 3d or 3 days\n"
            "• weekly or 1w or 1 week\n\n"
            "You can optionally add a specific time (if not specified, updates start from current time):\n"
            "• /set_interval 1 day 7AM\n"
            "• /set_interval 1d 7:00\n"
            "• /set_interval 12h 9:30"
        )
        return
    
    set_user_interval(user_id, interval_seconds, send_time)
    
    # Format interval for display
    if interval_seconds < 3600:
        display = f"{interval_seconds // 60} minutes"
    elif interval_seconds < 86400:
        display = f"{interval_seconds // 3600} hours"
    elif interval_seconds < 604800:
        display = f"{interval_seconds // 86400} day(s)"
    else:
        display = f"{interval_seconds // 604800} week(s)"
    
    time_notice = f" at {send_time}" if send_time else ""
    await update.message.reply_text(
        f"Interval set to {display}{time_notice}!\n\n"
        f"You will receive financial news updates every {display}{time_notice}."
    )

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /news command - send immediate news update."""
    chat_id = update.effective_chat.id
    
    # Optional acknowledgement to show the bot is working
    await update.message.reply_text("Fetching the latest financial news for you...")
    
    await send_financial_news(chat_id, context)

async def fetch_tavily_sources(query: str, max_results: int = 5) -> List[Dict]:
    """Search Tavily for general queries and return structured sources."""
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(TAVILY_API_URL, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])[:max_results]
    except Exception as e:
        return [{"title": "Error fetching sources", "url": "", "content": str(e)}]

async def clean_tavily_content(title: str, content: str) -> Tuple[str, str]:
    """Clean tavily search result to be readable by user."""
    prompt = f"""Summarize this:
                    {title}
                    {content}"""
    response = await run_llm(prompt)
    return response

async def run_llm(prompt: str) -> str:
    """Call OpenRouter API with the configured model."""
    import asyncio

    def _call() -> str:
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=400,
            temperature=0.6,
        )
        return completion.choices[0].message.content

    result = await asyncio.to_thread(_call)
    return result.strip()

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
            ("Economy", "latest economy overview today"),
            ("Global Stock Market", "latest global stock market indices and macroeconomic news today"),
            ("Individual Stocks", "latest individual company stock news today"),
            ("Crypto Market", "latest cryptocurrency market overview today"),
            ("Forex Market", "latest foreign exchange market overview today"),
            ("Precious Metals Market", "latest precious metal market overview today")
        ]

        sections: List[str] = []
        for label, query in categories:
            results = await search_financial_news(query, max_results=1)
            if results:
                item = results[0]
                raw_title = item.get("title", "No title")
                url = item.get("url", "")
                raw_content = item.get("content", "")
                
                # Clean the content using LLM
                clean_title, clean_content = await clean_tavily_content(raw_title, raw_content)
                
                sections.append(
                    f"{label}\n"
                    f"{clean_title}\n"
                    f"{clean_content}\n"
                    f"{url}\n"
                )
            else:
                sections.append(f"{label}\nNo news found for this category right now.\n")
        
        # Format message
        numbered_sections = []
        for idx, sec in enumerate(sections, start=1):
            numbered_sections.append(f"{idx}. {sec}")

        message = (
            "Financial News Update\n\n"
            + "\n".join(numbered_sections)
        )
        
        # Send message as plain text to avoid markdown parsing issues
        if len(message) > 4096:
            # Split message into chunks
            chunks = [message[i:i+4096] for i in range(0, len(message), 4096)]
            for chunk in chunks:
                await context.bot.send_message(chat_id=chat_id, text=chunk)
        else:
            await context.bot.send_message(chat_id=chat_id, text=message)
        
        # Update last sent time
        update_last_sent(chat_id, datetime.now().isoformat())
    
    except Exception as e:
        error_msg = f"Error sending news: {str(e)}"
        try:
            await context.bot.send_message(chat_id=chat_id, text=error_msg)
        except:
            print(f"Failed to send error message: {error_msg}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command - show user's current settings and next update time."""
    user_id = update.effective_user.id
    user_data = get_user_data(user_id)
    
    interval_seconds = user_data.get("interval_seconds")
    if not interval_seconds:
        await update.message.reply_text("No interval set. Use /set_interval to configure news updates.")
        return
    
    last_sent_str = user_data.get("last_sent")
    send_time_str = user_data.get("send_time")
    current_time = datetime.now()
    
    # Format interval for display
    if interval_seconds < 3600:
        display = f"{interval_seconds // 60} minutes"
    elif interval_seconds < 86400:
        display = f"{interval_seconds // 3600} hours"
    elif interval_seconds < 604800:
        display = f"{interval_seconds // 86400} day(s)"
    else:
        display = f"{interval_seconds // 604800} week(s)"
    
    # Calculate next update time
    if last_sent_str:
        try:
            last_sent = datetime.fromisoformat(last_sent_str)
            base_time = last_sent + timedelta(seconds=interval_seconds)
        except ValueError:
            base_time = current_time
    else:
        base_time = current_time
    
    due_time = base_time
    if send_time_str:
        due_time = align_to_send_time(base_time, send_time_str)
    
    time_notice = f" at {send_time_str}" if send_time_str else ""
    
    if current_time >= due_time:
        next_update = "Next update: Due now (will be sent within 5 minutes)"
    else:
        time_diff = due_time - current_time
        if time_diff.total_seconds() < 3600:
            next_update = f"Next update: In {int(time_diff.total_seconds() // 60)} minutes"
        elif time_diff.total_seconds() < 86400:
            next_update = f"Next update: In {int(time_diff.total_seconds() // 3600)} hours"
        else:
            next_update = f"Next update: {due_time.strftime('%Y-%m-%d %H:%M')}"
    
    last_sent_display = "Never" if not last_sent_str else datetime.fromisoformat(last_sent_str).strftime('%Y-%m-%d %H:%M')
    
    status_message = (
        f"Your News Settings\n\n"
        f"Interval: Every {display}{time_notice}\n"
        f"Last sent: {last_sent_display}\n"
        f"{next_update}\n\n"
        f"Use /news for immediate update\n"
        f"Use /set_interval to change settings"
    )
    
    await update.message.reply_text(status_message)

async def query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /query command - answer user question using LLM with Tavily sources."""
    if not context.args:
        await update.message.reply_text("Please provide a question, e.g. /query What is the bitcoin fear and greed index today?")
        return
    question = " ".join(context.args).strip()
    chat_id = update.effective_chat.id
    await update.message.reply_text("Searching for up-to-date sources...")

    sources = await fetch_tavily_sources(question, max_results=5)
    if not sources:
        await context.bot.send_message(chat_id=chat_id, text="No sources found at the moment. Please try again later.")
        return

    sources_text = []
    citations = []
    for idx, item in enumerate(sources, start=1):
        raw_title = item.get("title", "No title")
        url = item.get("url", "")
        raw_content = item.get("content", "")
        
        # Clean the content using LLM
        clean_title, clean_content = await clean_tavily_content(raw_title, raw_content)
        
        sources_text.append(f"[{idx}] {clean_title}\n{clean_content}\nURL: {url}")
        citations.append(f"{idx}. {clean_title} - {url}")

    prompt = (
        "You are FinNewsBot, a financial advisor. You only answer financial question, reject any unrelated question"
        "Use the numbered sources below to answer the user question. "
        "Cite sources inline with their bracket numbers (e.g., [1]). "
        "Only use information from the sources. If the answer isn't there, say you couldn't find it.\n\n"
        f"Question: {question}\n\n"
        "Sources:\n"
        + "\n\n".join(sources_text)
    )

    try:
        answer = await run_llm(prompt)
        if not answer:
            answer = "LLM returned an empty response."
    except Exception as e:
        answer = f"LLM processing error: {str(e)}"

    message = (
        f"Answer\n{answer}\n\n"
        "Sources\n" + "\n".join(citations)
    )

    await context.bot.send_message(chat_id=chat_id, text=message)

async def check_and_send_updates(context: ContextTypes.DEFAULT_TYPE):
    """Check all users and send updates if interval has passed."""
    users = get_all_users()
    current_time = datetime.now()
    
    print(f"Checking updates at {current_time}")  # Debug log
    
    for user_id, user_data in users:
        interval_seconds = user_data.get("interval_seconds")
        if not interval_seconds:
            print(f"User {user_id}: No interval set, skipping")
            continue
        
        last_sent_str = user_data.get("last_sent")
        send_time_str = user_data.get("send_time")
        
        if last_sent_str:
            try:
                last_sent = datetime.fromisoformat(last_sent_str)
                base_time = last_sent + timedelta(seconds=interval_seconds)
            except ValueError:
                print(f"User {user_id}: Invalid last_sent format, using current time")
                base_time = current_time - timedelta(seconds=interval_seconds + 1)
        else:
            # If no last_sent, send immediately
            print(f"User {user_id}: No last_sent, scheduling immediate send")
            base_time = current_time - timedelta(seconds=interval_seconds + 1)
        
        due_time = base_time
        if send_time_str:
            due_time = align_to_send_time(base_time, send_time_str)
        
        print(f"User {user_id}: Current={current_time}, Due={due_time}, Should send={current_time >= due_time}")
        
        if current_time < due_time:
            continue
        
        # Send update
        try:
            print(f"Sending news to user {user_id}")
            await send_financial_news(user_id, context)
            print(f"Successfully sent news to user {user_id}")
        except Exception as e:
            print(f"Error sending update to user {user_id}: {e}")

def main():
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    # Initialize database
    init_database()
    
    print("Bot configuration loaded successfully!")
    print(f"Using Tavily for news search and {MODEL_NAME} via OpenRouter for LLM answers.")
    print("SQLite database initialized for user data storage.")
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("set_interval", set_interval))
    application.add_handler(CommandHandler("news", news))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("query", query))
    
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
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()

