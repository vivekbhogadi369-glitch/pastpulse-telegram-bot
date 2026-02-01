import os
import logging
import tempfile
import asyncio

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

# Required env vars
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

SYSTEM_PROMPT = """
You are PastPulse AI (UPSC History tutor).

STRICT RULES:
- Answer ONLY from faculty documents retrieved via file_search.
- If answer not present, reply exactly:
Not found in faculty documents. Ask faculty to upload the relevant material.
- Do NOT guess. Do NOT use outside knowledge.

FORMAT:
- UPSC style headings + bullets
- Add Keywords (5–10)
- Add Timeline if relevant
- If MCQs: 4 options + answer + explanation
- Include 1–2 short quotes from documents.
""".strip()


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_TELEGRAM_IDS


# ---------------- Upload helpers ----------------
def upload_file_to_openai(local_path: str) -> str:
    """Upload file to OpenAI Files and return file_id."""
    with open(local_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="assistants")
    return uploaded.id


def attach_file_to_vector_store(file_id: str) -> None:
    """
    Attach an existing OpenAI file_id to the vector store.
    Works across multiple OpenAI SDK shapes.
    Raises Exception if not possible.
    """
    # Newer SDK: client.vector_stores.files.create(...)
    if hasattr(client, "vector_stores"):
        client.vector_stores.files.create(vector_store_id=VECTOR_STORE_ID, file_id=file_id)
        return

    # Some SDK variants may expose it under beta
    if hasattr(client, "beta") and hasattr(client.beta, "vector_stores"):
        client.beta.vector_stores.files.create(vector_store_id=VECTOR_STORE_ID, file_id=file_id)
        return

    # If neither exists, we can't attach programmatically
    raise RuntimeError(
        "Your OpenAI SDK does not support vector store attach (no vector_stores). "
        "Clear build cache deploy with openai==1.56.0 OR attach manually in OpenAI Dashboard."
    )


def upload_and_index_doc(local_path: str) -> str:
    """
    Full pipeline:
    1) upload file => file_id
    2) attach file_id to vector store (indexing)
    Returns file_id.
    """
    file_id = upload_file_to_openai(local_path)
    attach_file_to_vector_store(file_id)
    return file_id


# ---------------- Docs-only Answer (Assistants API) ----------------
def docs_only_answer_sync(user_text: str) -> str:
    assistant = client.beta.assistants.create(
        name="PastPulse Faculty Docs Only",
        model="gpt-4.1-mini",
        instructions=SYSTEM_PROMPT,
        tools=[{"type": "file_search"}],
        tool_resources={
            "file_search": {"vector_store_ids": [VECTOR_STORE_ID]}
        },
    )

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_text,
    )

    client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    text = messages.data[0].content[0].text.value.strip()

    if not text:
        return REFUSAL

    # Hard refusal gate: require quotes as evidence
    if ('"' not in text and "“" not in text and "”" not in text):
        return REFUSAL

    if "Not found in faculty documents" in text:
        return REFUSAL

    return text


async def docs_only_answer(user_text: str) -> str:
    try:
        return await asyncio.to_thread(docs_only_answer_sync, user_text)

    except APIConnectionError:
        return "⚠️ OpenAI connection error. Try again."

    except RateLimitError:
        return "⚠️ Too many requests. Try again after 1 minute."

    except APIStatusError as e:
        return f"⚠️ OpenAI API error: {e}"

    except Exception as e:
        logger.exception(e)
        return "Unexpected server error. Please try again."


# ---------------- Telegram Handlers ----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ PastPulse AI is Live!\n"
        "Ask any UPSC History question.\n\n"
        "Faculty Upload:\n"
        "/uploaddoc <ADMIN_SECRET>\n"
        "Then send PDF/DOC."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_text = update.message.text.strip()
    answer = await docs_only_answer(user_text)

    if len(answer) > 3900:
        answer = answer[:3900] + "…"

    await update.message.reply_text(answer)


async def uploaddoc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not ADMIN_SECRET or not ADMIN_TELEGRAM_IDS:
        await update.message.reply_text(
            "❌ Upload system not configured.\n"
            "Set ADMIN_SECRET and ADMIN_TELEGRAM_IDS in Render env vars."
        )
        return

    if not is_admin(user_id):
        await update.message.reply_text("❌ Not authorized.")
        return

    if len(context.args) != 1 or context.args[0] != ADMIN_SECRET:
        await update.message.reply_text("❌ Use:\n/uploaddoc <ADMIN_SECRET>")
        return

    context.user_data["awaiting_doc_upload"] = True
    await update.message.reply_text("✅ Now send the PDF/DOC file.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.document:
        return

    user_id = update.effective_user.id

    if not ADMIN_SECRET or not ADMIN_TELEGRAM_IDS:
        await update.message.reply_text(
            "❌ Upload system not configured.\n"
            "Set ADMIN_SECRET and ADMIN_TELEGRAM_IDS in Render env vars."
        )
        return

    if not is_admin(user_id):
        await update.message.reply_text("❌ Only faculty can upload.")
        return

    if not context.user_data.get("awaiting_doc_upload"):
        await update.message.reply_text("Send /uploaddoc <ADMIN_SECRET> first.")
        return

    doc = update.message.document
    file_obj = await context.bot.get_file(doc.file_id)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{doc.file_name or 'upload'}") as tmp:
            tmp_path = tmp.name

        await file_obj.download_to_drive(custom_path=tmp_path)

        await update.message.reply_text("⏳ Uploading & indexing...")

        file_id = await asyncio.to_thread(upload_and_index_doc, tmp_path)

        await update.message.reply_text(f"✅ Uploaded & indexed.\nFile ID: {file_id}")

    except Exception as e:
        logger.exception(e)
        # IMPORTANT: still show file_id if upload succeeded but attach failed
        msg = str(e)
        await update.message.reply_text(
            "❌ Upload failed.\n"
            f"{msg}\n\n"
            "If this says 'no vector_stores', do: Render → Manual Deploy → Clear build cache & deploy "
            "and ensure requirements has openai==1.56.0."
        )

    finally:
        context.user_data["awaiting_doc_upload"] = False
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled Telegram error", exc_info=context.error)


def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("uploaddoc", uploaddoc))

    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.add_error_handler(error_handler)

    logger.info("✅ PastPulse bot running (polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
