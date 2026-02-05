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

# This must match EXACTLY what you want users to see when content is missing.
REFUSAL = "Not found in faculty documents. Ask faculty to upload the relevant material."

# ---------------- MERGED SYSTEM PROMPT (GS-1 + UPSC Output Rules + Docs-Only) ----------------
SYSTEM_PROMPT = f"""
You are PastPulse AI — a strict, professional GS-1 History and Indian Art & Culture mentor for UPSC and State PSC preparation. Write in the style of an experienced UPSC examiner: crisp, analytical, syllabus-bound, and evidence-first.

========================
0) CLOSED-BOOK: FACULTY DOCS ONLY (HARD)
========================
You MUST use ONLY the retrieved faculty documents via file_search for factual content.
- Do NOT use outside knowledge.
- Do NOT guess or fill gaps.
- If the answer is not clearly supported by faculty documents, reply EXACTLY with:
{REFUSAL}

Evidence requirement:
- For normal Q&A and content answers, include at least ONE short direct quote (1–2 lines) from the retrieved faculty documents as evidence.

========================
1) SCOPE (STRICT GS-1 ONLY)
========================
Answer ONLY if the user’s query clearly falls within GS-1:
- Ancient Indian History
- Medieval Indian History
- Modern Indian History
- Prescribed World History themes
- Post-Independence history ONLY as historical processes (no current affairs)
- Indian Art & Culture

========================
2) HARD REFUSAL BOUNDARY (NON-NEGOTIABLE)
========================
Do NOT answer:
- Polity/Constitution/Governance, Economy, Agriculture, Social Justice, IR/Current Affairs,
  Geography/Environment, Science & Tech, Ethics/Internal Security/Essay

Mixed questions:
- Answer ONLY the historical/cultural part and refuse the rest in one line.

If the historical/cultural part is also not supported by faculty documents, respond EXACTLY with:
{REFUSAL}

========================
3) PRELIMS MCQs (UPSC STANDARD — STRICT)
========================
If the user asks for MCQs:
- Authentic UPSC Prelims standard (NOT school-level).
- Prefer statement-based / elimination-driven questions.
- Output MUST be:
  A) Questions (numbered)
  B) Answer Key (separate section) where EACH question has:
     - Correct option
     - 2–4 line justification
     - Elimination logic for EACH wrong option
- If unsupported by docs, reply EXACTLY: {REFUSAL}
- Include at least ONE short quote from documents.

========================
4) MAINS ANSWERS (MANDATORY ENRICHMENT PACK)
========================
If MAINS-style OR 150/250 words:
ALWAYS output:
1) Introduction
2) Body (analytical subheadings)
3) Conclusion
4) Chronological Timeline (5–10 bullets)
5) Conceptual Mindmap (ASCII)
6) High-Value Keywords (8–15)
7) PYQ Frequency Band: High/Medium/Low (do NOT invent years)

If unsupported by docs, reply EXACTLY: {REFUSAL}
Include at least ONE short quote from documents.

========================
5) QUALITY CONTROL
========================
- Examiner tone; no fluff.
- Prefer structured bullets.
- When unsure, exclude claims rather than guessing.
""".strip()


# ---------------- Utilities ----------------
def split_telegram_chunks(text: str, limit: int = 3900) -> List[str]:
    if len(text) <= limit:
        return [text]
    parts = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut == -1 or cut < 800:
            cut = remaining.rfind("\n", 0, limit)
        if cut == -1 or cut < 800:
            cut = limit
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_TELEGRAM_IDS


def is_mcq_request(user_text: str) -> bool:
    t = (user_text or "").lower()
    return any(k in t for k in ["mcq", "mcqs", "multiple choice", "prelims", "objective", "choose the correct"])


def is_mains_request(user_text: str) -> bool:
    t = (user_text or "").lower()
    if re.search(r"\b(150|250)\b", t):
        return True
    if "mains" in t:
        return True
    return any(k in t for k in ["discuss", "analyse", "analyze", "critically", "examine", "comment", "elucidate", "explain"])


def looks_like_mains_pack(text: str) -> bool:
    t = (text or "").lower()
    must = ["introduction", "conclusion", "timeline", "mindmap", "keywords", "pyq frequency"]
    return all(m in t for m in must)


# ---------------- Faculty upload helpers ----------------
def upload_file_to_openai(local_path: str) -> str:
    with open(local_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="assistants")
    return uploaded.id


def attach_file_to_vector_store(file_id: str) -> None:
    # Compatible with OpenAI SDK variations
    if hasattr(client, "vector_stores"):
        client.vector_stores.files.create(vector_store_id=VECTOR_STORE_ID, file_id=file_id)
        return
    if hasattr(client, "beta") and hasattr(client.beta, "vector_stores"):
        client.beta.vector_stores.files.create(vector_store_id=VECTOR_STORE_ID, file_id=file_id)
        return
    raise RuntimeError(
        "OpenAI SDK missing vector store attach. Clear build cache and ensure openai==1.56.0."
    )


def upload_and_index_doc(local_path: str) -> str:
    file_id = upload_file_to_openai(local_path)
    attach_file_to_vector_store(file_id)
    return file_id


# ---------------- Docs-only Q&A (Assistants + file_search) ----------------
_ASSISTANT_ID = None


def _get_or_create_assistant_id() -> str:
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
    logger.info("Created Assistant: ASSISTANT_ID=%s VECTOR_STORE_ID=%s", _ASSISTANT_ID, VECTOR_STORE_ID)
    return _ASSISTANT_ID


def _assistant_run(user_content: str) -> str:
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
    return messages.data[0].content[0].text.value.strip() if messages.data else ""


def wrap_user_query(user_text: str) -> str:
    # Hard-pin the output rules each time, so it doesn’t drift to “normal answers”.
    return f"""
Follow these operating rules STRICTLY:
1) Only GS-1 History + Indian Art & Culture.
2) Use ONLY faculty documents via file_search. No outside knowledge.
3) If not supported by faculty documents, reply EXACTLY: {REFUSAL}
4) For factual/content answers, include at least ONE short direct quote (1–2 lines) from the faculty documents.
5) If the user asks MCQs: UPSC prelims level, statement-based, and provide elimination logic for each wrong option in Answer Key.
6) If MAINS-style or 150/250 words: MUST output Introduction, Body (subheadings), Conclusion, Timeline, ASCII Mindmap, Keywords, PYQ Frequency Band (no invented years).

User question:
{user_text}
""".strip()


def docs_only_answer_sync(user_text: str) -> str:
    wrapped = wrap_user_query(user_text)
    text = _assistant_run(wrapped)

    if not text:
        return REFUSAL
    if REFUSAL in text:
        return REFUSAL

    # Must contain quote to prove retrieval for content answers/MCQs/mains
    has_quote = ('"' in text) or ("“" in text) or ("”" in text)
    if not has_quote:
        return REFUSAL

    # If MAINS requested but enrichment pack missing, force a reformat pass
    if (not is_mcq_request(user_text)) and is_mains_request(user_text):
        if not looks_like_mains_pack(text):
            reform = f"""
Reformat into the mandated UPSC MAINS ENRICHMENT PACK:
- Introduction
- Body (analytical subheadings)
- Conclusion
- Chronological Timeline (5–10 bullets)
- Conceptual Mindmap (ASCII)
- High-Value Keywords (8–15)
- PYQ Frequency Band: High/Medium/Low (do NOT invent years)

Critical:
- Do NOT add new facts beyond what is supported by faculty documents.
- Include at least ONE short direct quote (1–2 lines).
- If unsupported, reply EXACTLY: {REFUSAL}

TEXT TO REFORMAT:
{text}
""".strip()
            text2 = _assistant_run(reform)
            if text2 and (REFUSAL not in text2):
                has_quote2 = ('"' in text2) or ("“" in text2) or ("”" in text2)
                if has_quote2:
                    return text2
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
        "Scope: GS-1 History + Indian Art & Culture (Docs-only).\n\n"
        "What it can do:\n"
        "• UPSC-standard MAINS answers (150/250 words + timeline/mindmap/keywords)\n"
        "• UPSC-standard Prelims MCQs (statement-based + elimination logic)\n\n"
        "Note: Answer evaluation is currently disabled.\n\n"
        "Faculty Upload:\n"
        "/uploaddoc <ADMIN_SECRET>\n"
        "Then send PDF/DOC."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    raw_text = update.message.text.strip()

    # Evaluation disabled (explicit)
    if re.search(r"\bevaluate\s*my\s*answer\b", raw_text, flags=re.IGNORECASE):
        await update.message.reply_text(
            "⚠️ Answer evaluation is currently disabled.\n"
            "Ask for MAINS (150/250 words) or UPSC Prelims MCQs instead."
        )
        return

    answer = await docs_only_answer(raw_text)
    for part in split_telegram_chunks(answer):
        await update.message.reply_text(part)


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
    await update.message.reply_text("✅ Now send the faculty PDF/DOC file.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.document:
        return

    user_id = update.effective_user.id
    doc = update.message.document

    tmp_path = None
    try:
        file_obj = await context.bot.get_file(doc.file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{doc.file_name or 'upload'}") as tmp:
            tmp_path = tmp.name
        await file_obj.download_to_drive(custom_path=tmp_path)

        # Faculty upload only
        if context.user_data.get("awaiting_doc_upload"):
            if not is_admin(user_id):
                await update.message.reply_text("❌ Only faculty can upload.")
                return

            await update.message.reply_text("⏳ Uploading & indexing faculty document...")
            file_id = await asyncio.to_thread(upload_and_index_doc, tmp_path)

            global _ASSISTANT_ID
            _ASSISTANT_ID = None  # refresh retrieval

            await update.message.reply_text(f"✅ Uploaded & indexed.\nFile ID: {file_id}")
            return

        # If not in upload mode, tell user evaluation is disabled & PDFs not accepted from students.
        await update.message.reply_text(
            "⚠️ Student answer evaluation is currently disabled.\n"
            "If you meant to upload faculty material, faculty should run /uploaddoc <ADMIN_SECRET> first."
        )

    except Exception as e:
        logger.exception(e)
        await update.message.reply_text("❌ File handling failed. Please try again.")
    finally:
        if context.user_data.get("awaiting_doc_upload"):
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

    # Keep document handler only for faculty uploads
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.add_error_handler(error_handler)

    logger.info("✅ PastPulse bot running (polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
