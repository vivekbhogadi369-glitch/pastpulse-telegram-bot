import os
import logging
import asyncio

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError


# ---------- Logging ----------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("pastpulse-bot")


# ---------- Env ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment variables.")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are PastPulse AI, a UPSC History tutor. "
    "Answer clearly with UPSC-friendly structure, crisp points, and factual accuracy. "
    "If the question is vague, ask 1 short clarification."
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()  # safe default


# ---------- OpenAI helper ----------
async def call_openai(user_text: str) -> str:
    # Simple retry for temporary network errors
    for attempt in range(3):
        try:
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.4,
                max_tokens=500,
            )
            return resp.choices[0].message.content.strip()

        except (APIConnectionError,) as e:
            logger.warning(f"OpenAI connection error attempt {attempt+1}/3: {e}")
            await asyncio.sleep(2 * (attempt + 1))

        except (RateLimitError,) as e:
            logger.warning(f"OpenAI rate limit: {e}")
            return "⚠️ OpenAI is rate-limiting right now. Please retry in 30 seconds."

        except (APIStatusError,) as e:
            logger.error(f"OpenAI API status error: {e}")
            return "⚠️ OpenAI server error. Please retry in 30 seconds."

        except Exception as e:
            logger.exception(f"Unexpected OpenAI error: {e}")
            return "⚠️ Something went wrong on the server. Please try again."

    return "⚠️ OpenAI connection problem from server. Please retry in 30 seconds."


# ---------- Telegram handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("✅ PastPulse AI is Live!\nAsk any UPSC History question.")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send any UPSC History question.\n\nExamples:\n- Write about Ashoka in 10 lines\n- Causes of 1857 revolt\n- Mauryan administration"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if not text:
        return

    # quick typing indicator
    try:
        await update.message.chat.send_action("typing")
    except Exception:
        pass

    answer = await call_openai(text)
    await update.message.reply_text(answer)


def main() -> None:
    logger.info("Starting bot...")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Important for Render worker: long polling
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
