import os
import re
import logging
import tempfile
import asyncio
from typing import List

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError


# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Env Vars ----------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
VECTOR_STORE_ID = os.environ.get("VECTOR_STORE_ID", "").strip()

ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "").strip()
ADMIN_TELEGRAM_IDS_RAW = os.environ.get("ADMIN_TELEGRAM_IDS", "").strip()

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in Render env vars.")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in Render env vars.")
if not VECTOR_STORE_ID:
    raise RuntimeError("Missing VECTOR_STORE_ID in Render env vars.")

ADMIN_TELEGRAM_IDS = set(
    int(x.strip())
    for x in ADMIN_TELEGRAM_IDS_RAW.split(",")
    if x.strip().isdigit()
)

client = OpenAI(api_key=OPENAI_API_KEY)

REFUSAL = "Not found in faculty documents. Ask faculty to upload the relevant material."


# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = f"""
You are PastPulse AI — a strict GS-1 History and Indian Art & Culture mentor for UPSC and State PSC preparation.

========================
DOCS ONLY RULE
========================
Use ONLY faculty documents via file_search.
If not supported, reply EXACTLY:
{REFUSAL}

Include at least ONE short quote from documents.

========================
MCQs
========================
UPSC Prelims level only.
Statement-based.
Elimination logic mandatory.

========================
MAINS
========================
Always include:
Introduction
Body
Conclusion
Timeline
Mindmap
Keywords
PYQ frequency band
""".strip()


# ---------------- Utilities ----------------
def split_telegram_chunks(text: str, limit: int = 3900) -> List[str]:
    if len(text) <= limit:
        return [text]
    parts = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = limit
        parts.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        parts.append(remaining)
    return parts


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_TELEGRAM_IDS


# ---------------- OpenAI Assistants ----------------
_ASSISTANT_ID = None


def _get_or_create_assistant_id():
    global _ASSISTANT_ID
    if _ASSISTANT_ID:
        return _ASSISTANT_ID

    assistant = client.beta.assistants.create(
        name="PastPulse Faculty Docs Only",
        model="gpt-4.1",
        instructions=SYSTEM_PROMPT,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}},
    )
    _ASSISTANT_ID = assistant.id
    return _ASSISTANT_ID


def _assistant_run(user_content: str):
    assistant_id = _get_or_create_assistant_id()

    thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_content,
    )

    client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    return messages.data[0].content[0].text.value.strip()


async def docs_only_answer(user_text: str) -> str:
    try:
        return await asyncio.to_thread(_assistant_run, user_text)
    except Exception:
        return "Error generating answer."


# ---------------- Telegram Handlers ----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ PastPulseAI is Live!\n"
        "Ask any History/Indian Art & Culture questions for UPSC, State PSCs, SI & Constable, SSC-CGL, Banking and many more exams."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    raw_text = update.message.text.strip()

    answer = await docs_only_answer(raw_text)

    for part in split_telegram_chunks(answer):
        await update.message.reply_text(part)


async def uploaddoc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not is_admin(user_id):
        await update.message.reply_text("Not authorized.")
        return

    if len(context.args) != 1 or context.args[0] != ADMIN_SECRET:
        await update.message.reply_text("Use: /uploaddoc <ADMIN_SECRET>")
        return

    context.user_data["awaiting_doc_upload"] = True
    await update.message.reply_text("Send faculty PDF/DOC.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("awaiting_doc_upload"):
        return

    user_id = update.effective_user.id
    if not is_admin(user_id):
        return

    doc = update.message.document
    file_obj = await context.bot.get_file(doc.file_id)

    tmp = tempfile.NamedTemporaryFile(delete=False)
    await file_obj.download_to_drive(custom_path=tmp.name)

    file_id = client.files.create(file=open(tmp.name, "rb"), purpose="assistants").id

    client.vector_stores.files.create(
        vector_store_id=VECTOR_STORE_ID,
        file_id=file_id
    )

    await update.message.reply_text("Faculty document uploaded.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception(context.error)


def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("uploaddoc", uploaddoc))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.add_error_handler(error_handler)

    app.run_polling()


if __name__ == "__main__":
    main()
