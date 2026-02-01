import os
import logging
import asyncio

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI
from openai import APIConnectionError, AuthenticationError, RateLimitError, APIStatusError


# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("pastpulse-bot")


# -----------------------------
# ENV VARS (Render -> Environment)
# -----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN in environment variables.")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")


# -----------------------------
# OPENAI CLIENT
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

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
- Keywords
- Timeline where relevant
- Keep answers crisp and exam-oriented
"""


# -----------------------------
# OPENAI CALL (run in a thread so Telegram async doesn't freeze)
# -----------------------------
def ask_gpt_sync(user_text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_text.strip()},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


async def ask_gpt(user_text: str) -> str:
    return await asyncio.to_thread(ask_gpt_sync, user_text)


# -----------------------------
# TELEGRAM HANDLERS
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ PastPulse AI is Live!\nAsk any UPSC History question.")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send any UPSC History question.\nExample: 'Write about Ashoka in 10 lines'."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    try:
        reply = await ask_gpt(user_text)
        await update.message.reply_text(reply)

    except AuthenticationError:
        await update.message.reply_text(
            "❌ OpenAI API key problem.\nCheck OPENAI_API_KEY in Render → Environment."
        )

    except RateLimitError:
        await update.message.reply_text(
            "⚠️ OpenAI rate limit hit. Please retry after 30–60 seconds."
        )

    except APIConnectionError:
        await update.message.reply_text(
            "⚠️ OpenAI connection problem from server. Please retry in 30 seconds."
        )

    except APIStatusError as e:
        await update.message.reply_text(
            f"⚠️ OpenAI server error ({getattr(e, 'status_code', 'unknown')}). Try again in 30 seconds."
        )

    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        await update.message.reply_text("⚠️ Something went wrong. Please try again.")


# -----------------------------
# MAIN
# -----------------------------
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started. Polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
