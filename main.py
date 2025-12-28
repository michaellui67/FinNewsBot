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

def get_user_timezone_display(user_id: int) -> str:
    """Get user's timezone in a user-friendly display format."""
    tz_str = get_user_timezone(user_id)
    
    # Convert Etc/GMT format back to user-friendly format
    if tz_str.startswith('Etc/GMT'):
        try:
            offset = int(tz_str.replace('Etc/GMT', ''))
            # Etc/GMT signs are reversed, so flip them back
            if offset == 0:
                return 'UTC'
            elif offset > 0:
                return f'UTC-{offset}'
            else:
                return f'UTC+{abs(offset)}'
        except:
            return tz_str
    
    # Convert common timezone names to UTC offset format
    timezone_to_utc = {
        'Asia/Jakarta': 'UTC+7',
        'Asia/Bangkok': 'UTC+7',
        'Asia/Ho_Chi_Minh': 'UTC+7',
        'Asia/Tokyo': 'UTC+9',
        'Asia/Shanghai': 'UTC+8',
        'Asia/Kolkata': 'UTC+5:30',
        'Asia/Dubai': 'UTC+4',
        'Europe/London': 'UTC+0',  # GMT
        'Europe/Berlin': 'UTC+1',
        'Europe/Paris': 'UTC+1',
        'Europe/Rome': 'UTC+1',
        'Europe/Madrid': 'UTC+1',
        'Europe/Moscow': 'UTC+3',
        'US/Eastern': 'UTC-5',  # EST
        'US/Pacific': 'UTC-8',  # PST
        'US/Central': 'UTC-6',  # CST
        'US/Mountain': 'UTC-7', # MST
        'Australia/Sydney': 'UTC+11',
        'America/Sao_Paulo': 'UTC-3',
    }
    
    if tz_str in timezone_to_utc:
        return timezone_to_utc[tz_str]
    
    # For other timezone names, try to get the current offset
    try:
        tz = pytz.timezone(tz_str)
        now = datetime.utcnow()
        utc_dt = pytz.utc.localize(now)
        local_dt = utc_dt.astimezone(tz)
        offset = local_dt.utcoffset()
        
        if offset:
            total_seconds = int(offset.total_seconds())
            hours = total_seconds // 3600
            minutes = abs(total_seconds % 3600) // 60
            
            if minutes == 0:
                return f'UTC{hours:+d}'
            else:
                sign = '+' if hours >= 0 else '-'
                return f'UTC{sign}{abs(hours)}:{minutes:02d}'
    except:
        pass
    
    return tz_str

def parse_interval(interval_str: str) -> Optional[int]:
    """Parse interval string and return seconds. Supports dynamic numbers."""
    interval_str = interval_str.lower().strip()
    
    # Handle special aliases first
    aliases = {
        'daily': 24 * 3600,
        'weekly': 7 * 24 * 3600,
        'hourly': 3600,
    }
    
    if interval_str in aliases:
        return aliases[interval_str]
    
    # Parse dynamic intervals like "5h", "2 hours", "10 days", "3 weeks"
    import re
    
    # Pattern to match: number + unit
    # Examples: "5h", "2 hours", "10 days", "3 weeks", "1d", "24hours"
    pattern = r'^(\d+)\s*(h|hour|hours|d|day|days|w|week|weeks)$'
    match = re.match(pattern, interval_str)
    
    if match:
        number = int(match.group(1))
        unit = match.group(2)
        
        # Validate reasonable limits
        if unit in ['h', 'hour', 'hours']:
            if number < 1 or number > 168:  # 1 hour to 1 week in hours
                return None
            return number * 3600
        elif unit in ['d', 'day', 'days']:
            if number < 1 or number > 30:  # 1 day to 30 days
                return None
            return number * 24 * 3600
        elif unit in ['w', 'week', 'weeks']:
            if number < 1 or number > 4:  # 1 week to 4 weeks
                return None
            return number * 7 * 24 * 3600
    
    return None

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
            result = f"Etc/GMT{-hours if sign == '+' else hours:+d}"
            print(f"Timezone parsing: '{tz_input}' -> '{result}'")
            return result
        else:
            # For non-hour offsets, create a fixed offset name
            result = f"UTC{sign}{hours:02d}:{minutes:02d}"
            print(f"Timezone parsing: '{tz_input}' -> '{result}'")
            return result
    
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
        f"• /set_interval <frequency> [time] <timezone> - Set to periodically receive news\n"
    )
    
    await update.message.reply_text(welcome_message)

async def set_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /set_interval with optional timezone setting."""
    user_id = update.effective_user.id
    
    if not context.args:
        current_tz = get_user_timezone(user_id)
        user_time = get_user_time(user_id)
        
        await update.message.reply_text(
            f"Set your news update interval with timezone.\n\n"
            f"Usage: /set_interval <frequency> [time] <timezone>\n\n"
            f"Examples:\n"
            f"• /set_interval daily UTC+7 (will use current time)\n"
            f"• /set_interval 8h 9AM Asia/Bangkok\n"
            f"• /set_interval weekly 15:00 EST\n"
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
    
    # Validate interval using dynamic parser
    interval_seconds = parse_interval(interval_str)
    if interval_seconds is None:
        await update.message.reply_text(
            f"Invalid interval: {interval_str}\n\n"
            f"Valid formats:\n"
            f"• Hours: 1h, 5h, 24h, 2 hours\n"
            f"• Days: 1d, 7d, 30d, 3 days\n"
            f"• Weeks: 1w, 4w, 2 weeks\n"
            f"• daily, weekly, hourly\n\n"
            f"Examples:\n"
            f"• /set_interval daily UTC+7 (will use current time)\n"
            f"• /set_interval 8h 9AM Asia/Bangkok\n"
            f"• /set_interval weekly 15:00 EST\n"
        )
        return
    
    # Timezone is mandatory
    if not timezone_input:
        await update.message.reply_text(
            f"Timezone is required!\n\n"
            f"Usage: /set_interval <frequency> [time] <timezone>\n\n"
            f"Valid timezone formats:\n"
            f"• UTC+7, GMT-5, +8, -3\n"
            f"• Asia/Bangkok, Europe/London\n"
            f"• EST, PST, CET, JST\n\n"
            f"Examples:\n"
            f"• /set_interval daily UTC+7 (will use current time)\n"
            f"• /set_interval 8h 9AM Asia/Bangkok\n"
            f"• /set_interval weekly 15:00 EST\n"
        )
        return
    
    # Validate and set timezone
    parsed_tz = parse_timezone_input(timezone_input)
    if not parsed_tz:
        await update.message.reply_text(
            f"Invalid timezone: {timezone_input}\n\n"
            f"Valid formats:\n"
            f"• UTC+7, GMT-5, +8, -3\n"
            f"• Asia/Bangkok, Europe/London\n"
            f"• EST, PST, CET, JST\n\n"
            f"Examples:\n"
            f"• /set_interval daily UTC+7 (will use current time)\n"
            f"• /set_interval 8h 9AM Asia/Bangkok\n"
            f"• /set_interval weekly 15:00 EST\n"
        )
        return
    
    tz_success = set_user_timezone(user_id, parsed_tz)
    if not tz_success:
        await update.message.reply_text("Error setting timezone. Please try again.")
        return
    print(f"User {user_id} timezone set to: {parsed_tz}")
    print(f"Parsed timezone input '{timezone_input}' -> stored as '{parsed_tz}'")
    
    # Save interval
    success = set_user_interval(user_id, interval_seconds, send_time)
    if not success:
        await update.message.reply_text(
            "Sorry, there was an error saving your settings. Please try again."
        )
        return
    
    # Show confirmation - get timezone after it's been set
    user_tz = get_user_timezone(user_id)
    user_tz_display = get_user_timezone_display(user_id)
    
    # Get current UTC time and convert to user's timezone
    current_time_utc = datetime.utcnow()
    user_current_time = get_user_time(user_id, current_time_utc)
    
    print(f"User {user_id} confirmation - stored tz: {user_tz}, display tz: {user_tz_display}")
    print(f"UTC time: {current_time_utc}, User time: {user_current_time}")
    
    # Format interval for display
    if interval_seconds < 3600:
        display = f"{interval_seconds // 60} minutes"
    elif interval_seconds < 86400:
        display = f"{interval_seconds // 3600} hours"
    elif interval_seconds < 604800:
        display = f"{interval_seconds // 86400} day(s)"
    else:
        display = f"{interval_seconds // 604800} week(s)"
    
    time_notice = f" at {send_time} {user_tz_display}" if send_time else ""
    
    # Calculate next update time using the same UTC reference
    if send_time:
        next_update_utc = align_to_send_time_with_tz(current_time_utc, send_time, user_id)
        next_update_user_tz = get_user_time(user_id, next_update_utc)
        next_update_str = f"{next_update_user_tz.strftime('%Y-%m-%d %H:%M')} {user_tz_display}"
    else:
        next_update_user_tz = user_current_time + timedelta(seconds=interval_seconds)
        if interval_seconds < 3600:
            next_update_str = f"in {interval_seconds // 60} minutes"
        elif interval_seconds < 86400:
            next_update_str = f"in {interval_seconds // 3600} hours"
        else:
            next_update_str = f"{next_update_user_tz.strftime('%Y-%m-%d %H:%M')} {user_tz_display}"
    
    confirmation_message = (
        f"News updates configured successfully!\n\n"
        f"Frequency: Every {display}{time_notice}\n"
        f"Current time: {user_current_time.strftime('%Y-%m-%d %H:%M:%S')} {user_tz_display}\n"
        f"Next update: {next_update_str}\n\n"
        f"Other commands:\n"
        f"• /news - Get immediate update\n"
        f"• /status - Check settings\n"
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
    user_tz_display = get_user_timezone_display(user_id)
    
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
    time_notice = f" at {send_time_str} {user_tz_display}" if send_time_str else ""
    
    if current_time_utc >= due_time_utc:
        next_update = "Next update: Due now (will be sent within 5 minutes)"
    else:
        time_diff = due_time_utc - current_time_utc
        if time_diff.total_seconds() < 3600:
            next_update = f"Next update: In {int(time_diff.total_seconds() // 60)} minutes"
        elif time_diff.total_seconds() < 86400:
            next_update = f"Next update: In {int(time_diff.total_seconds() // 3600)} hours"
        else:
            next_update = f"Next update: {due_time_user.strftime('%Y-%m-%d %H:%M')} {user_tz_display}"
    
    status_message = (
        f"Your News Settings\n\n"
        f"Interval: Every {display}{time_notice}\n"
        f"Current time: {current_time_user.strftime('%Y-%m-%d %H:%M:%S')} {user_tz_display}\n"
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
    current_tz_display = get_user_timezone_display(user_id)
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
            new_tz_display = get_user_timezone_display(user_id)
            await update.message.reply_text(
                f"Timezone updated!\n"
                f"Current time: {new_user_time.strftime('%Y-%m-%d %H:%M:%S')} {new_tz_display}"
            )
        else:
            await update.message.reply_text("Error setting timezone. Please try again.")
    else:
        # Show current timezone
        await update.message.reply_text(
            f"Current time: {user_time.strftime('%Y-%m-%d %H:%M:%S')} {current_tz_display}\n\n"
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

