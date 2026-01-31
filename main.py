import os
import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError


logging.basicConfig(level=logging.INFO)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN env var")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY env var")

# OpenAI client with retries + timeout (prevents bot from dying on network hiccups)
client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=30,
    max_retries=3,
)

SYSTEM_PROMPT = """
You are PastPulse AI, a UPSC History mentor.

Teach:
- Ancient History
- Medieval History
- Modern History
- World History
- Art & Culture

Style Rules:
- UPSC-ready format
- Headings + bullet points
- Timeline + keywords when relevant
- If MCQs asked: 4 options + answer + explanation
"""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ PastPulse AI is Live!\nAsk any UPSC History question.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        )

        reply = resp.choices[0].message.content.strip()
        if not reply:
            reply = "I couldn't generate a response. Please ask again."
        await update.message.reply_text(reply)

    except APIConnectionError:
        await update.message.reply_text(
            "⚠️ OpenAI connection problem from server. Please retry in 30 seconds."
        )

    except RateLimitError:
        await update.message.reply_text(
            "⚠️ Rate limit hit. Please retry after 1 minute."
        )

    except APIStatusError as e:
        await update.message.reply_text(
            f"⚠️ OpenAI error: {e.status_code}. Please retry."
        )

    except Exception as e:
        logging.exception("Unexpected error")
        await update.message.reply_text(f"⚠️ Unexpected error: {type(e).__name__}")


def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("Bot started. Polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
