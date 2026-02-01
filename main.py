import os
import logging
import tempfile
import asyncio

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

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

# Hard required env vars (bot must not start without them)
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment variables.")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment variables.")
if not VECTOR_STORE_ID:
    raise RuntimeError("Missing VECTOR_STORE_ID in environment variables.")

# Admin vars: required if you want uploads to work securely
if not ADMIN_SECRET:
    logger.warning("ADMIN_SECRET is missing. Faculty upload command will NOT work securely.")
if not ADMIN_TELEGRAM_IDS_RAW:
    logger.warning("ADMIN_TELEGRAM_IDS is missing. No one will be authorized to upload documents.")

ADMIN_TELEGRAM_IDS = set(
    int(x.strip())
    for x in ADMIN_TELEGRAM_IDS_RAW.split(",")
    if x.strip().isdigit()
)

client = OpenAI(api_key=OPENAI_API_KEY)

REFUSAL = "Not found in faculty documents. Ask faculty to upload the relevant material."

SYSTEM_PROMPT = """
You are PastPulse AI (UPSC History tutor).

STRICT RULES (NON-NEGOTIABLE):
- Answer ONLY from faculty documents retrieved via file_search.
- If the answer is not present in the documents, reply exactly:
Not found in faculty documents. Ask faculty to upload the relevant material.
- Do NOT use outside knowledge. Do NOT guess.

FORMAT (UPSC-ready):
- Use headings + bullet points
- If user asks “150 words” or “250 words”, follow it
- Add: Keywords (5–10)
- Add: Timeline (5–8 points) if relevant
- If user asks MCQs: give 4 options + answer + explanation
- Include 1–3 short quotes from the documents as evidence.
""".strip()

# ---------------- Admin helpers ----------------
def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_TELEGRAM_IDS


def attach_doc_to_vector_store(local_path: str) -> str:
    """
    Upload file to OpenAI Files (purpose=assistants) and attach to vector store.
    Returns uploaded OpenAI file_id.
    """
    with open(local_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="assistants")

    client.vector_stores.files.create(
        vector_store_id=VECTOR_STORE_ID,
        file_id=uploaded.id
    )
    return uploaded.id


# ---------------- Docs-only answer ----------------
def docs_only_answer_sync(user_text: str) -> str:
    """
    OpenAI Responses API + file_search bound to VECTOR_STORE_ID.
    IMPORTANT: Use the correct syntax (NO tool_resources kwarg).
    Hard-gate: if the model doesn't quote docs, refuse (prevents hallucination).
    """

    resp = client.responses.create(
        model="gpt-4.1-mini",
        instructions=SYSTEM_PROMPT,
        input=user_text,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
        }],
    )

    text = (resp.output_text or "").strip()
    if not text:
        return REFUSAL

    # HARD GATE: No quote marks => refuse (forces evidence)
    if ('"' not in text and "“" not in text and "”" not in text and "‘" not in text and "’" not in text):
        return REFUSAL

    # If model tries to include refusal inside longer text, force exact refusal
    if "Not found in faculty documents" in text:
        return REFUSAL

    return text


async def docs_only_answer(user_text: str) -> str:
    """
    Run the sync OpenAI call in a thread (keeps Telegram async loop responsive).
    Retries for transient errors.
    """
    for attempt in range(3):
        try:
            return await asyncio.to_thread(docs_only_answer_sync, user_text)

        except APIConnectionError as e:
            logger.warning(f"OpenAI connection error attempt {attempt+1}/3: {e}")
            await asyncio.sleep(2 * (attempt + 1))

        except RateLimitError as e:
            logger.warning(f"Rate limited attempt {attempt+1}/3: {e}")
            await asyncio.sleep(3 * (attempt + 1))

        except APIStatusError as e:
            logger.error(f"OpenAI API status error: {getattr(e, 'status_code', 'unknown')} | {str(e)}")
            return "OpenAI API error. Please check model access and VECTOR_STORE_ID."

        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return "Unexpected server error. Please try again."

    return "OpenAI connection problem from server. Please retry in 30 seconds."


# ---------------- Telegram Handlers ----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ PastPulse AI is Live!\nAsk any UPSC History question.\n\n"
        "Faculty: use /uploaddoc <ADMIN_SECRET> then send PDF/DOC to add notes."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()

    if len(user_text) < 2:
        await update.message.reply_text("Please type a proper question 🙂")
        return

    answer = await docs_only_answer(user_text)

    # Telegram message limit safety (~4096). Keep buffer.
    if len(answer) > 3900:
        answer = answer[:3900] + "…"

    await update.message.reply_text(answer)


async def uploaddoc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if not ADMIN_SECRET or not ADMIN_TELEGRAM_IDS:
        await update.message.reply_text(
            "❌ Upload system is not configured.\n"
            "Set ADMIN_SECRET and ADMIN_TELEGRAM_IDS in Render env variables."
        )
        return

    if not is_admin(user_id):
        await update.message.reply_text("❌ You are not authorized to upload documents.")
        return

    # /uploaddoc <ADMIN_SECRET>
    if len(context.args) != 1 or context.args[0] != ADMIN_SECRET:
        await update.message.reply_text("❌ Use:\n/uploaddoc <ADMIN_SECRET>\nThen send the PDF/DOC.")
        return

    context.user_data["awaiting_doc_upload"] = True
    await update.message.reply_text("✅ Now send the PDF/DOC file. I will add it to the faculty knowledge base.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.document:
        return

    user_id = update.effective_user.id

    if not ADMIN_SECRET or not ADMIN_TELEGRAM_IDS:
        await update.message.reply_text(
            "❌ Upload system is not configured.\n"
            "Set ADMIN_SECRET and ADMIN_TELEGRAM_IDS in Render env variables."
        )
        return

    if not is_admin(user_id):
        await update.message.reply_text("❌ Only faculty/admin can upload documents.")
        return

    if not context.user_data.get("awaiting_doc_upload"):
        await update.message.reply_text("Send /uploaddoc <ADMIN_SECRET> first, then send the file.")
        return

    doc = update.message.document
    filename = (doc.file_name or "").lower()
    allowed = (".pdf", ".doc", ".docx", ".txt")

    if filename and not filename.endswith(allowed):
        context.user_data["awaiting_doc_upload"] = False
        await update.message.reply_text("❌ Please upload PDF/DOC/DOCX/TXT only.")
        return

    file_obj = await context.bot.get_file(doc.file_id)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{doc.file_name or 'upload'}") as tmp:
            tmp_path = tmp.name

        await file_obj.download_to_drive(custom_path=tmp_path)

        await update.message.reply_text("⏳ Uploading to faculty knowledge base…")

        file_id = await asyncio.to_thread(attach_doc_to_vector_store, tmp_path)

        await update.message.reply_text(f"✅ Uploaded & indexed.\nFile ID: {file_id}")

    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        await update.message.reply_text(f"❌ Upload failed: {e}")

    finally:
        context.user_data["awaiting_doc_upload"] = False
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Prevent silent crashes; keeps details in Render logs
    logger.exception("Unhandled Telegram error", exc_info=context.error)


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("uploaddoc", uploaddoc))

    # Faculty upload handler (documents)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Student Q&A (text)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Error handler (so bot doesn't die silently)
    app.add_error_handler(error_handler)

    logger.info("✅ Bot started successfully (polling)...")

    # IMPORTANT: clears old queued updates and reduces weirdness after redeploy
    # NOTE: This does NOT fix Conflict if another instance is running elsewhere.
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
