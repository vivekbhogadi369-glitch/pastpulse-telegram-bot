import os
import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI

logging.basicConfig(level=logging.INFO)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


# ✅ Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi Vivek Sir 👋\nI am PastPulse AI 🤖\nAsk any UPSC History question!"
    )


# ✅ Handle user messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a UPSC History mentor."},
            {"role": "user", "content": user_text},
        ],
    )

    reply = response.choices[0].message.content
    await update.message.reply_text(reply)


def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
