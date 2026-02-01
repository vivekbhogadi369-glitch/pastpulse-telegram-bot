import os
import logging
import asyncio

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from openai import AsyncOpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError


# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("pastpulse-bot")


# ---------- Env ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment variables.")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment variables.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=30.0, max_retries=2)


SYSTEM_PROMPT = """
You are PastPulse AI — a UPSC History mentor.
Write answers in a UPSC topper-notes style: crisp, structured, factual, exam-focused.

Rules:
1) Prefer bullet points. Use short lines.
2) Always include: Keywords + Timeline (when relevant) + 1–2 PYQ-style angles.
3) If user asks "10 lines", give exactly 10 numbered lines.
4) For normal questions: use headings:
   - Context
   - Core Points
   - Keywords (5–8)
   - PYQ Link (1–2 lines)
   - Quick Revision (2–3 lines)
5) Do NOT hallucinate specific inscription numbers/dates unless confident. If unsure, say "approx." or "noted in inscriptions".
6) If question is vague, ask 1 clarifying line at the end, but still give a useful answer first.
7) Keep language simple, exam-ready, no fluff.
"""


# ---------- Helpers ----------
async def ask_openai(user_text: str) -> str:
    """
    Calls OpenAI with retries for transient connection issues.
    """
    # Small manual retry loop (in addition to SDK retries)
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()

        except APIConnectionError as e:
            logger.warning(f"OpenAI connection error attempt {attempt+1}/3: {e}")
            await asyncio.sleep(2 * (attempt + 1))

        except RateLimitError as e:
            logger.warning(f"Rate limited attempt {attempt+1}/3: {e}")
            await asyncio.sleep(3 * (attempt + 1))

        except APIStatusError as e:
            # Non-connection errors with status code (e.g., 401/403/400)
            logger.error(f"OpenAI API status error: {e.status_code} | {e.message}")
            return "OpenAI API error (status). Please check API key/model and try again."

        except Exception as e:
            logger.exception(f"Unexpected OpenAI error: {e}")
            return "Unexpected server error. Please try again."

    return "OpenAI connection problem from server. Please retry in 30 seconds."


# ---------- Telegram Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ PastPulse AI is Live!\nAsk any UPSC History question."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()

    # Optional: ignore very short spam
    if len(user_text) < 2:
        await update.message.reply_text("Please type a proper question 🙂")
        return

    answer = await ask_openai(user_text)
    # Telegram message limit safety
    if len(answer) > 3500:
        answer = answer[:3500] + "..."

    await update.message.reply_text(answer)


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started. Polling...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
