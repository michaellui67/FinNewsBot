import os
import json
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from huggingface_hub import InferenceClient
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "deepseek-ai/DeepSeek-R1")

# Initialize Tavily client (will be set in main())
tavily_client = None

# Initialize Hugging Face Inference client (will be set in main())
hf = None

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

def set_user_interval(user_id: int, interval_seconds: int):
    """Set user's news interval."""
    data = load_user_data()
    if str(user_id) not in data:
        data[str(user_id)] = {}
    data[str(user_id)]["interval_seconds"] = interval_seconds
    data[str(user_id)]["last_sent"] = None
    save_user_data(data)

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
    
    # Parse interval
    interval_str = " ".join(context.args).lower()
    interval_seconds = None
    
    interval_map = {
        "3h": 3 * 3600, "3 hours": 3 * 3600, "3hour": 3 * 3600,
        "6h": 6 * 3600, "6 hours": 6 * 3600, "6hour": 6 * 3600,
        "12h": 12 * 3600, "12 hours": 12 * 3600, "12hour": 12 * 3600,
        "1d": 24 * 3600, "1 day": 24 * 3600, "1day": 24 * 3600,
        "3d": 3 * 24 * 3600, "3 days": 3 * 24 * 3600, "3day": 3 * 24 * 3600,
        "1w": 7 * 24 * 3600, "1 week": 7 * 24 * 3600, "1week": 7 * 24 * 3600,
    }
    
    interval_seconds = interval_map.get(interval_str)
    
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
    
    set_user_interval(user_id, interval_seconds)
    
    # Format interval for display
    if interval_seconds < 3600:
        display = f"{interval_seconds // 60} minutes"
    elif interval_seconds < 86400:
        display = f"{interval_seconds // 3600} hours"
    elif interval_seconds < 604800:
        display = f"{interval_seconds // 86400} days"
    else:
        display = f"{interval_seconds // 604800} week(s)"
    
    await update.message.reply_text(
        f"✅ Interval set to {display}!\n\n"
        f"You will receive financial news updates every {display}.\n"
        f"Your first update will be sent shortly."
    )
    
    # Send first news update immediately
    await send_financial_news(update.message.chat_id, context)

async def search_financial_news(query: str = "latest stock market cryptocurrency news") -> str:
    """Search for financial news using Tavily."""
    try:
        # Search for financial news
        search_results = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=False
        )
        
        # Format results
        news_items = []
        if search_results.get("answer"):
            news_items.append(f"📊 {search_results['answer']}\n")
        
        if search_results.get("results"):
            news_items.append("📰 Top News:\n")
            for i, result in enumerate(search_results["results"][:5], 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                content = result.get("content", "")[:200]  # First 200 chars
                news_items.append(f"{i}. {title}\n{content}...\n🔗 {url}\n")
        
        return "\n".join(news_items) if news_items else "No financial news found at the moment."
    
    except Exception as e:
        return f"Error searching for news: {str(e)}"

async def process_with_llm(query: str, context: str) -> str:
    """Process query with Hugging Face LLM using Inference API."""
    try:
        # Combine query and context
        prompt = f"Based on this financial news context: {context[:500]}\n\nSummarize the key points about: {query}"
        
        # Generate response using Inference API
        response = hf.text_generation(
            prompt,
            max_new_tokens=200,
            temperature=0.7,
            return_full_text=False
        )
        
        summary = response.strip()
        return summary[:300]  # Limit to 300 characters
    
    except Exception as e:
        return f"LLM processing error: {str(e)}"

async def send_financial_news(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Send financial news to user."""
    try:
        # Search for financial news
        news_content = await search_financial_news("latest stock market cryptocurrency news today")
        
        # Process with LLM for summary
        llm_summary = await process_with_llm(
            "financial markets and cryptocurrency",
            news_content
        )
        
        # Format message
        message = (
            "📈 **Financial News Update** 📈\n\n"
            f"{news_content}\n\n"
            f"🤖 **AI Summary:**\n{llm_summary}"
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
        if last_sent_str:
            last_sent = datetime.fromisoformat(last_sent_str)
            time_since_last = (current_time - last_sent).total_seconds()
            if time_since_last < interval_seconds:
                continue
        else:
            # First time, send immediately
            pass
        
        # Send update
        try:
            await send_financial_news(int(user_id_str), context)
        except Exception as e:
            print(f"Error sending update to user {user_id_str}: {e}")

def main():
    """Start the bot."""
    global tavily_client, hf
    
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found in environment variables")
    
    # Initialize Tavily client
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    
    # Initialize Hugging Face Inference client
    print("Initializing Hugging Face Inference client...")
    hf = InferenceClient(token=HF_TOKEN, model=HF_MODEL_NAME)
    print("Hugging Face client initialized successfully!")
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("set_interval", set_interval))
    
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

