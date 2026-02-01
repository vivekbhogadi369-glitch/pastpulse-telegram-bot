import os
import logging
import asyncio

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APITimeoutError, APIStatusError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pastpulse")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are PastPulse AI, a UPSC History mentor.\n"
    "Teach: Ancient, Medieval, Modern, World History, Art & Culture.\n"
    "Style: UPSC-ready, headings + bullet points, crisp, factual.\n"
    "If asked for MCQs: provide answers + brief explanation.\n"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ PastPulse AI is Live!\nAsk any UPSC History question.")

async def ask_openai(user_text: str) -> str:
    # Retry a few times for flaky network (Render sometimes does this)
    for attempt in range(1, 4):
        try:
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.4,
                max_tokens=700,
            )
            return resp.choices[0].message.content.strip()

        except (APIConnectionError, APITimeoutError) as e:
            logger.warning(f"OpenAI connection/timeout (attempt {attempt}): {e}")
            await asyncio.sleep(2 * attempt)

        except RateLimitError as e:
            logger.warning(f"OpenAI rate limit (attempt {attempt}): {e}")
            await asyncio.sleep(5 * attempt)

        except APIStatusError as e:
            logger.error(f"OpenAI API status error: {e}")
            return "⚠️ OpenAI server issue. Try again in 30 seconds."

        except Exception as e:
            logger.exception(f"Unexpected OpenAI error: {e}")
            return "⚠️ Something broke on server. Try again in 30 seconds."

    return "⚠️ OpenAI connection problem from server. Please retry in 30 seconds."

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return

    await update.message.chat.send_action(action="typing")
    answer = await ask_openai(text)
    # Telegram message limit safety
    if len(answer) > 3800:
        answer = answer[:3800] + "\n\n(…trimmed)"
    await update.message.reply_text(answer)

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started. Polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
