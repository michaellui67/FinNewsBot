import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import httpx
from openai import OpenAI
import pytz
import re

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
            send_time TEXT,
            timezone TEXT DEFAULT 'UTC'
        )
    ''')
    
    # Add timezone column to existing tables if it doesn't exist
    try:
        cursor.execute('ALTER TABLE user_settings ADD COLUMN timezone TEXT DEFAULT "UTC"')
    except sqlite3.OperationalError:
        # Column already exists
        pass
    
    conn.commit()
    conn.close()

def auto_detect_timezone(user_id: int, update: Update) -> str:
    """
    Return user's current timezone or default to UTC.
    No automatic detection - just ensures user has a timezone set.
    """
    current_tz = get_user_timezone(user_id)
    return current_tz

def get_user_timezone(user_id: int) -> str:
    """Get user's timezone."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT timezone FROM user_settings WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 'UTC'

def set_user_timezone(user_id: int, timezone_str: str) -> bool:
    """Set user's timezone."""
    try:
        # Validate timezone using pytz
        pytz.timezone(timezone_str)
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO user_settings (user_id, timezone) VALUES (?, ?)
        ''', (user_id, timezone_str))
        cursor.execute('''
            UPDATE user_settings SET timezone = ? WHERE user_id = ?
        ''', (timezone_str, user_id))
        conn.commit()
        conn.close()
        return True
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"Invalid timezone: {timezone_str}")
        return False
    except Exception as e:
        print(f"Error setting timezone: {e}")
        return False

def parse_timezone_input(tz_input: str) -> Optional[str]:
    """Parse timezone input from user using pytz."""
    tz_input = tz_input.strip()
    
    # Common timezone mappings
    timezone_aliases = {
        'utc': 'UTC',
        'gmt': 'GMT',
        'est': 'US/Eastern',
        'pst': 'US/Pacific',
        'cst': 'US/Central',
        'mst': 'US/Mountain',
        'jst': 'Asia/Tokyo',
        'cet': 'Europe/Berlin',
        'ist': 'Asia/Kolkata',
        'bst': 'Europe/London',
        'aest': 'Australia/Sydney',
    }
    
    # Check if it's a common alias
    if tz_input.lower() in timezone_aliases:
        return timezone_aliases[tz_input.lower()]
    
    # Check if it's a UTC offset like +7, -5, UTC+7, GMT-5
    utc_offset_pattern = r'^(UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2}))?$'
    match = re.match(utc_offset_pattern, tz_input.upper())
    if match:
        sign = match.group(2)
        hours = int(match.group(3))
        minutes = int(match.group(4)) if match.group(4) else 0
        
        if hours > 14 or (hours == 14 and minutes > 0):
            return None  # Invalid offset
        
        # Use pytz's fixed offset timezone
        from datetime import timezone, timedelta
        offset_seconds = (hours * 3600 + minutes * 60)
        if sign == '-':
            offset_seconds = -offset_seconds
        
        # For simple hour offsets, use Etc/GMT format (note: signs are reversed in Etc/GMT)
        if minutes == 0:
            return f"Etc/GMT{-hours if sign == '+' else hours:+d}"
        else:
            # For non-hour offsets, create a fixed offset name
            return f"UTC{sign}{hours:02d}:{minutes:02d}"
    
    # Try as direct timezone name - pytz will validate it
    try:
        pytz.timezone(tz_input)
        return tz_input
    except pytz.exceptions.UnknownTimeZoneError:
        return None

def get_user_time(user_id: int, dt: Optional[datetime] = None) -> datetime:
    """Get current time in user's timezone."""
    if dt is None:
        dt = datetime.utcnow()
    
    user_tz_str = get_user_timezone(user_id)
    try:
        user_tz = pytz.timezone(user_tz_str)
        # Use pytz's localize for naive datetimes, astimezone for aware ones
        if dt.tzinfo is None:
            utc_dt = pytz.utc.localize(dt)
        else:
            utc_dt = dt
        return utc_dt.astimezone(user_tz)
    except pytz.exceptions.UnknownTimeZoneError:
        # Fallback to UTC
        return pytz.utc.localize(dt) if dt.tzinfo is None else dt

def align_to_send_time_with_tz(reference: datetime, send_time_str: str, user_id: int) -> datetime:
    """Align the reference datetime to the next occurrence of the preferred send time in user's timezone."""
    user_tz_str = get_user_timezone(user_id)
    try:
        user_tz = pytz.timezone(user_tz_str)
        
        # Convert reference to user timezone
        if reference.tzinfo is None:
            reference = pytz.utc.localize(reference)
        user_time = reference.astimezone(user_tz)
        
        # Parse send time
        hour, minute = map(int, send_time_str.split(":"))
        
        # Create target time in user timezone using pytz's localize
        target_date = user_time.date()
        naive_target = datetime.combine(target_date, datetime.min.time().replace(hour=hour, minute=minute))
        candidate = user_tz.localize(naive_target)
        
        if candidate <= user_time:
            # Move to next day
            next_day = target_date + timedelta(days=1)
            naive_target = datetime.combine(next_day, datetime.min.time().replace(hour=hour, minute=minute))
            candidate = user_tz.localize(naive_target)
        
        # Convert back to UTC for storage
        return candidate.astimezone(pytz.utc).replace(tzinfo=None)
    except pytz.exceptions.UnknownTimeZoneError:
        # Fallback to original function
        return align_to_send_time(reference, send_time_str)

def get_user_interval(user_id: int) -> int:
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT interval_seconds FROM user_settings WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def set_user_interval(user_id: int, interval_seconds: int, send_time: Optional[str] = None):
    """Set user's news interval and optional preferred send time."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_settings (user_id, interval_seconds, send_time)
            VALUES (?, ?, ?)
        ''', (user_id, interval_seconds, send_time))
        
        conn.commit()
        conn.close()
        print(f"Successfully saved interval for user {user_id}: {interval_seconds}s, send_time: {send_time}")
        return True
    except Exception as e:
        print(f"Error saving user interval: {e}")
        return False

def get_user_data(user_id: int) -> Dict:
    """Get all user data."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT interval_seconds, send_time, timezone FROM user_settings WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "interval_seconds": result[0],
            "send_time": result[1],
            "timezone": result[2] if result[2] else 'UTC'
        }
    return {}

def get_all_users() -> List[Tuple[int, Dict]]:
    """Get all users and their settings."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, interval_seconds, send_time, timezone FROM user_settings')
    results = cursor.fetchall()
    conn.close()
    
    users = []
    for row in results:
        user_data = {
            "interval_seconds": row[1],
            "send_time": row[2],
            "timezone": row[3] if row[3] else 'UTC'
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

def parse_interval_time_timezone(raw_input: str) -> Dict:
    """
    Parse interval, time, and timezone from user input.
    Returns dict with interval, time, timezone, and error fields.
    """
    tokens = raw_input.strip().split()
    if not tokens:
        return {"interval": "", "time": None, "timezone": None, "error": "No input provided"}
    
    interval = tokens[0]
    remaining_tokens = tokens[1:]
    time_str = None
    timezone_str = None
    
    # Try to identify time and timezone from remaining tokens
    for i, token in enumerate(remaining_tokens):
        # Check if token looks like a time (contains : or ends with AM/PM)
        if ':' in token or token.lower().endswith(('am', 'pm')):
            if time_str is None:
                parsed_time = parse_time_input(token)
                if parsed_time:
                    time_str = parsed_time
                    continue
        
        # Check if token looks like a timezone
        if parse_timezone_input(token):
            timezone_str = token
            continue
        
        # If we haven't found time yet, try parsing as time
        if time_str is None:
            parsed_time = parse_time_input(token)
            if parsed_time:
                time_str = parsed_time
                continue
        
        # Try remaining tokens as timezone (could be multi-word like "Asia/Bangkok")
        remaining = " ".join(remaining_tokens[i:])
        if parse_timezone_input(remaining):
            timezone_str = remaining
            break
    
    return {
        "interval": interval,
        "time": time_str,
        "timezone": timezone_str,
        "error": None
    }

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
        f"Hello {username}!\n\n"
        f"Welcome to FinNewsBot!\n\n"
        f"I provide automated financial news updates covering stocks, crypto, forex, and market analysis.\n\n"
        f"Available commands:\n"
        f"• /news - Get news immediately\n"
        f"• /status - Check your current settings\n"
        f"• /query <question> - Ask a financial question\n"
        f"• /set_interval <frequency> [time] [timezone] - Set to periodically receive news\n"
    )
    
    await update.message.reply_text(welcome_message)

async def set_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /set_interval command with optional timezone setting."""
    user_id = update.effective_user.id
    
    if not context.args:
        current_tz = get_user_timezone(user_id)
        user_time = get_user_time(user_id)
        
        await update.message.reply_text(
            f"Set your news update interval.\n\n"
            f"Current time: {user_time.strftime('%Y-%m-%d %H:%M:%S ')}{current_tz}\n\n"
            f"Usage: /set_interval <frequency> [time] [timezone]\n\n"
            f"Examples:\n"
            f"• /set_interval 3h (will automatically use current time and timezone)\n"
            f"• /set_interval weekly 3PM (will automatically use current timezone)\n"
            f"• /set_interval daily 15:00 UTC+7\n"
        )
        return
    
    # Parse the full input: interval, time, and timezone
    raw_input = " ".join(context.args)
    parsed_result = parse_interval_time_timezone(raw_input)
    
    if parsed_result["error"]:
        await update.message.reply_text(parsed_result["error"])
        return
    
    interval_str = parsed_result["interval"]
    send_time = parsed_result["time"]
    timezone_input = parsed_result["timezone"]
    
    # Validate interval
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
            f"Invalid interval: {interval_str}\n\n"
            f"Example on how to set intervals:\n"
            f"• /set_interval 3h (will automatically use current time and timezone)\n"
            f"• /set_interval weekly 3PM (will automatically use current timezone)\n"
            f"• /set_interval daily 15:00 UTC+7\n"
        )
        return
    
    # Set timezone if provided, otherwise try to auto-detect
    if timezone_input:
        parsed_tz = parse_timezone_input(timezone_input)
        if not parsed_tz:
            await update.message.reply_text(
                f"Invalid timezone: {timezone_input}\n\n"
                f"Valid formats:\n"
                f"• UTC+7, GMT-5, +8, -3\n"
                f"• Asia/Bangkok, Europe/London\n"
                f"• EST, PST, CET, JST\n\n"
                f"Examples:\n"
                f"• /set_interval daily 9:00 UTC+7\n"
                f"• /set_interval weekly 15:30 Asia/Bangkok\n"
                f"• /set_interval 12h EST"
            )
            return
        
        tz_success = set_user_timezone(user_id, parsed_tz)
        if not tz_success:
            await update.message.reply_text("Error setting timezone. Please try again.")
            return
    else:
        # No timezone provided - user will use their current timezone (default UTC)
        current_tz = get_user_timezone(user_id)
        if current_tz == 'UTC':
            print(f"User {user_id} using default timezone: UTC")
    
    # Save interval
    success = set_user_interval(user_id, interval_seconds, send_time)
    if not success:
        await update.message.reply_text(
            "Sorry, there was an error saving your settings. Please try again."
        )
        return
    
    # Show confirmation
    user_tz = get_user_timezone(user_id)
    user_current_time = get_user_time(user_id)
    
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
    
    # Calculate next update time
    current_time_utc = datetime.utcnow()
    if send_time:
        next_update_utc = align_to_send_time_with_tz(current_time_utc, send_time, user_id)
        next_update_user_tz = get_user_time(user_id, next_update_utc)
        next_update_str = next_update_user_tz.strftime('%Y-%m-%d %H:%M')
    else:
        next_update_user_tz = user_current_time + timedelta(seconds=interval_seconds)
        if interval_seconds < 3600:
            next_update_str = f"in {interval_seconds // 60} minutes"
        elif interval_seconds < 86400:
            next_update_str = f"in {interval_seconds // 3600} hours"
        else:
            next_update_str = next_update_user_tz.strftime('%Y-%m-%d %H:%M')
    
    confirmation_message = (
        f"News updates configured successfully!\n\n"
        f"Frequency: Every {display}{time_notice}\n"
        f"Current time: {user_current_time.strftime('%Y-%m-%d %H:%M:%S ')}{user_tz}\n"
        f"Next update: {next_update_str}\n\n"
        f"Commands:\n"
        f"• /news - Get immediate update\n"
        f"• /status - Check settings\n"
    )
    
    await update.message.reply_text(confirmation_message)
        )
        return
    
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
    
    # Calculate next update time for confirmation using user's timezone
    current_time = datetime.utcnow()
    user_current_time = get_user_time(user_id, current_time)
    user_tz = get_user_timezone(user_id)
    
    if send_time:
        # If specific time is set, calculate next occurrence in user's timezone
        next_update_utc = align_to_send_time_with_tz(current_time, send_time, user_id)
        next_update_user_tz = get_user_time(user_id, next_update_utc)
        next_update_str = next_update_user_tz.strftime('%Y-%m-%d %H:%M')
    else:
        # If no specific time, next update is after the interval
        next_update_user_tz = user_current_time + timedelta(seconds=interval_seconds)
        if interval_seconds < 3600:
            next_update_str = f"in {interval_seconds // 60} minutes"
        elif interval_seconds < 86400:
            next_update_str = f"in {interval_seconds // 3600} hours"
        else:
            next_update_str = next_update_user_tz.strftime('%Y-%m-%d %H:%M')
    
    confirmation_message = (
        f"News interval successfully configured!\n\n"
        f"Frequency: Every {display}{time_notice}\n"
        f"Current time: {user_current_time.strftime('%Y-%m-%d %H:%M:%S ')}{user_tz}\n"
        f"Next update: {next_update_str}\n"
        f"Content: Financial news covering economy, stocks, crypto, forex, and precious metals\n\n"
        f"You can:\n"
        f"• Use /news for immediate update\n"
        f"• Use /status to check your settings\n"
        f"• Use /timezone to change your timezone\n"
        f"• Use /set_interval again to change settings"
    )
    
    await update.message.reply_text(confirmation_message)

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
    prompt = f"""Summarize without greetings:
                    {title}
                    {content}"""
    
    try:
        response = await run_llm(prompt)
        return title, response
        
    except Exception as e:
        print(f"Error cleaning content with LLM: {e}")
        # Simple fallback cleaning
        clean_title = title.split('|')[0].strip() if '|' in title else title
        clean_title = clean_title.strip('"\'""''')
        clean_content = content[:200] + "..." if len(content) > 200 else content
        return clean_title, clean_content

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
    
    send_time_str = user_data.get("send_time")
    user_tz = get_user_timezone(user_id)
    
    current_time_utc = datetime.utcnow()
    current_time_user = get_user_time(user_id, current_time_utc)
    
    # Format interval for display
    if interval_seconds < 3600:
        display = f"{interval_seconds // 60} minutes"
    elif interval_seconds < 86400:
        display = f"{interval_seconds // 3600} hours"
    elif interval_seconds < 604800:
        display = f"{interval_seconds // 86400} day(s)"
    else:
        display = f"{interval_seconds // 604800} week(s)"
    
    # Calculate next update time using timezone-aware functions
    base_time = current_time_utc
    
    due_time_utc = base_time
    if send_time_str:
        due_time_utc = align_to_send_time_with_tz(base_time, send_time_str, user_id)
    else:
        due_time_utc = base_time + timedelta(seconds=interval_seconds)
    
    due_time_user = get_user_time(user_id, due_time_utc)
    time_notice = f" at {send_time_str}" if send_time_str else ""
    
    if current_time_utc >= due_time_utc:
        next_update = "Next update: Due now (will be sent within 5 minutes)"
    else:
        time_diff = due_time_utc - current_time_utc
        if time_diff.total_seconds() < 3600:
            next_update = f"Next update: In {int(time_diff.total_seconds() // 60)} minutes"
        elif time_diff.total_seconds() < 86400:
            next_update = f"Next update: In {int(time_diff.total_seconds() // 3600)} hours"
        else:
            next_update = f"Next update: {due_time_user.strftime('%Y-%m-%d %H:%M')}"
    
    status_message = (
        f"Your News Settings\n\n"
        f"Interval: Every {display}{time_notice}\n"
        f"Current time: {current_time_user.strftime('%Y-%m-%d %H:%M:%S ')}{user_tz}\n"
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

async def timezone_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /timezone command - view or set timezone."""
    user_id = update.effective_user.id
    current_tz = get_user_timezone(user_id)
    user_time = get_user_time(user_id)
    
    if context.args:
        # Set timezone
        timezone_input = " ".join(context.args)
        parsed_tz = parse_timezone_input(timezone_input)
        
        if not parsed_tz:
            await update.message.reply_text(
                f"Invalid timezone: {timezone_input}\n\n"
                f"Use /set_interval to set timezone with your schedule.\n"
                f"Example: /set_interval daily 9:00 UTC+7"
            )
            return
        
        success = set_user_timezone(user_id, parsed_tz)
        if success:
            new_user_time = get_user_time(user_id)
            await update.message.reply_text(
                f"Timezone updated!\n"
                f"Current time: {new_user_time.strftime('%Y-%m-%d %H:%M:%S ')}{parsed_tz}"
            )
        else:
            await update.message.reply_text("Error setting timezone. Please try again.")
    else:
        # Show current timezone
        await update.message.reply_text(
            f"Current time: {user_time.strftime('%Y-%m-%d %H:%M:%S ')}{current_tz}\n\n"
            f"To change timezone: /timezone UTC+7\n"
            f"Or use: /set_interval daily 9:00 Asia/Bangkok"
        )

async def check_and_send_updates(context: ContextTypes.DEFAULT_TYPE):
    """Check all users and send updates if interval has passed."""
    users = get_all_users()
    current_time_utc = datetime.utcnow()
    
    print(f"Checking updates at {current_time_utc} UTC")  # Debug log
    
    for user_id, user_data in users:
        interval_seconds = user_data.get("interval_seconds")
        if not interval_seconds:
            print(f"User {user_id}: No interval set, skipping")
            continue
        
        send_time_str = user_data.get("send_time")
        user_tz = get_user_timezone(user_id)
        user_current_time = get_user_time(user_id, current_time_utc)
        
        # Calculate when next update should be sent
        base_time = current_time_utc
        
        due_time_utc = base_time
        if send_time_str:
            due_time_utc = align_to_send_time_with_tz(base_time, send_time_str, user_id)
        else:
            # For interval-based updates without specific time, send immediately
            due_time_utc = current_time_utc
        
        due_time_user = get_user_time(user_id, due_time_utc)
        
        print(f"User {user_id} ({user_tz}): Current={user_current_time.strftime('%H:%M')}, Due={due_time_user.strftime('%H:%M')}, Should send={current_time_utc >= due_time_utc}")
        
        if current_time_utc < due_time_utc:
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
    application.add_handler(CommandHandler("timezone", timezone_cmd))
    
    # Start periodic task
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(
            check_and_send_updates,
            interval=300,  # Check every 5 minutes
            first=10  # Start after 10 seconds
        )
    
    print("Bot is starting...")
    
    try:
        # Run bot with error handling
        application.run_polling(
            allowed_updates=Update.ALL_TYPES, 
            drop_pending_updates=True,
            close_loop=False
        )
    except Exception as e:
        if "Conflict" in str(e) and "getUpdates" in str(e):
            print("\nERROR: Another bot instance is already running!")
            print("Please stop all other instances of this bot before starting a new one.")
        else:
            print(f"\nUnexpected error: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {e}")
        exit(1)

