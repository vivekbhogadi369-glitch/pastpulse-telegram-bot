import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from openai import OpenAI

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN env var")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY env var")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are PastPulse AI, a UPSC History mentor.

Teach:
- Ancient History
- Medieval History
- Modern History
- World History
- Art & Culture

Style Rules:
- UPSC-ready format
- Headings + bullet points
- Add keywords
- Be concise and clear
"""

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ PastPulse AI is Live!\nAsk any UPSC History question.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send any UPSC History question. Example: 'Explain Ashoka in 10 lines'")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            timeout=30
        )
        answer = resp.choices[0].message.content
        await update.message.reply_text(answer)

    except Exception as e:
        logging.exception("OpenAI error")
        await update.message.reply_text(
            "⚠️ OpenAI connection problem from server.\nPlease retry in 30 seconds."
        )

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("Bot started. Polling...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
