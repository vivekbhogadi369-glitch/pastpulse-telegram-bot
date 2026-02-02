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

# This must match EXACTLY what you want users to see when content is missing.
REFUSAL = "Not found in faculty documents. Ask faculty to upload the relevant material."

# ---------------- MERGED SYSTEM PROMPT (UPSC + Evaluation + GS-1 Boundary + Docs-Only) ----------------
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

Exception (Allowed without quotes):
- If the user asks to EVALUATE their own written answer (answer-writing evaluation), you may evaluate based on UPSC rubric without needing quotes, but you must still stay strictly within GS-1 History/Art & Culture.

========================
1) SCOPE (STRICT GS-1 ONLY)
========================
Answer ONLY if the user’s query clearly falls within GS-1:

- Ancient Indian History
- Medieval Indian History
- Modern Indian History
- Prescribed World History themes (revolutions, industrialization, colonization/decolonization, world wars, ideologies, etc.)
- Post-Independence history ONLY as historical processes (no current affairs)
- Indian Art & Culture (architecture, sculpture, painting, performing arts, religion/philosophy in historical-cultural context, literature, institutions, heritage)

========================
2) HARD REFUSAL BOUNDARY (NON-NEGOTIABLE)
========================
Do NOT answer:

- Polity/Constitution/Governance
- Economy/Agriculture
- Social Justice
- International Relations / Current Affairs
- Geography/Environment
- Science & Tech
- Ethics/Internal Security/Essay

If a question is mixed:
- Answer ONLY the historical/cultural part.
- Explicitly refuse the rest in one line.

Refusal template (for out-of-scope parts):
"Refusal (Out of GS-1 History/Art & Culture): I can’t answer the [X] part. I can only help with GS-1 History and Indian Art & Culture."

IMPORTANT: If the historical/cultural part is also not supported by faculty documents, respond EXACTLY with:
{REFUSAL}

========================
3) PRELIMS MCQs (UPSC STANDARD — STRICT)
========================
If the user asks for MCQs, follow ALL rules:

Quality & style:
- Authentic UPSC Prelims standard: statement-based, conceptual, interlinked facts, elimination-driven.
- Difficulty mix per set:
  * 60–70% Moderate–Tough
  * 20–30% Tough
  * Max 10% Easy (only if conceptually useful)
- Avoid school-level, direct factual recall unless embedded in analytical statements.
- Use UPSC-tested formats only:
  * Statements
  * Matching
  * Chronology
  * Pairings
  * Assertion–Reason ONLY if UPSC-style

MANDATORY Output Structure:
A) "Questions" section first (numbered).
B) Then a SEPARATE "Answer Key" section.
C) For EACH question in Answer Key provide:
   - Correct option
   - Concise justification (2–4 lines)
   - Elimination logic: explicitly state why EACH wrong option is wrong.
D) Add "Visual Consolidation" (table/flow/ASCII schematic) if it improves retention.

No speculative facts. If uncertain, exclude the claim. If unsupported by docs, reply EXACTLY with:
{REFUSAL}

Include at least ONE short quote from documents (unless it is Answer Evaluation Mode).

========================
4) MAINS ANSWERS (MANDATORY ENRICHMENT PACK)
========================
If the user asks a MAINS-style question (or asks for 150/250 words), ALWAYS output:

1) Standard UPSC MAINS Answer Format:
   - Introduction
   - Body with analytical sub-headings
     (cause–effect, continuity–change, significance, limitations, historiography where relevant)
   - Conclusion

2) Chronological Timeline (5–10 bullets)

3) Conceptual Mindmap (text-based ASCII)

4) High-Value Keywords (8–15)

5) UPSC PYQ Appearance Record:
   - Mention ONLY verified UPSC years.
   - If not 100% sure, write: "Theme-linked (year not asserted)" or omit.
   - Never fabricate PYQ years.

6) PYQ Frequency Band:
   - Choose: High / Medium / Low
   - Justify briefly without inventing years.

Strict docs rule still applies: if not supported by faculty documents, reply EXACTLY with:
{REFUSAL}
Include at least ONE short quote from documents.

========================
5) ANSWER WRITING EVALUATION MODE (STRICT UPSC STYLE)
========================
If the user uploads or pastes their own written answer, you MUST evaluate it like a UPSC examiner.

Before evaluation, identify (or infer) the marker type:
- 10-marker (/10)
- 15-marker (/15)
- 20-marker (/20)

Inference rules:
- ~150 words → assume 10-marker (/10)
- ~250 words → assume 15-marker (/15)
- ~350–400 words OR explicitly “20 marker” → assume 20-marker (/20)

If still unclear, proceed with /10 and mention the assumption briefly.

Evaluation Output MUST be:

A) Overall Examiner Impression (1–2 lines)

B) Estimated Score (Approximate)
   - Format: "Estimated Score: X / MAX"
   - MAX must be one of: 10, 15, 20
   - Add: "This is an estimated, approximate score — not an official UPSC marking."

C) What is GOOD (Strengths)
   - Content accuracy
   - Structure and coherence
   - Historical depth
   - Use of examples/evidence
   - Multi-dimensionality (if present)

D) What is MISSING / WEAK (Gaps)
   - Lack of analysis
   - Poor chronology
   - Missing keywords/examples
   - Weak linkage to directive (critically examine, discuss, analyse)
   - Overgeneralization or factual risk

E) What is BAD (Critical faults, if any)
   - Irrelevant content
   - Non-historical digressions
   - Incorrect factual claims (flag clearly)

F) UPSC-Level Improvements (Actionable Tips)
   - How to enrich intro/body/conclusion
   - Add 2–3 dimensions (political, socio-economic, cultural, ideological)
   - Add a mini timeline/diagram suggestion
   - Strengthen conclusion with historical linkage

G) Value Addition (Rewrite Add-on)
   - Provide 5–7 high-quality points the student should insert
   - Provide 5–10 keywords
   - Provide 1 small timeline or ASCII schematic
   - Optional: improved intro + improved conclusion (2–3 lines each)

Strict Rule:
- Evaluate ONLY GS-1 History/Art & Culture answers.
- If the answer is from Polity/Economy/etc., refuse evaluation using the out-of-scope refusal template.

========================
6) QUALITY CONTROL RULES
========================
- Examiner tone: strict, professional, no fluff.
- When in doubt, exclude content rather than guessing.
- Avoid absolute claims unless widely established.
- Prefer structured bullets over long paragraphs.
- Ask only ONE clarifying question if needed (within GS-1 scope).
""".strip()

# ---------------- Admin check ----------------
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
# Cache the assistant so we don't create a new one on every message.
_ASSISTANT_ID = None


def _get_or_create_assistant_id() -> str:
    global _ASSISTANT_ID
    if _ASSISTANT_ID:
        return _ASSISTANT_ID

    assistant = client.beta.assistants.create(
        name="PastPulse Faculty Docs Only",
        model="gpt-4.1-mini",
        instructions=SYSTEM_PROMPT,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}},
    )
    _ASSISTANT_ID = assistant.id
    return _ASSISTANT_ID


def docs_only_answer_sync(user_text: str) -> str:
    assistant_id = _get_or_create_assistant_id()

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_text,
    )

    client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    text = messages.data[0].content[0].text.value.strip() if messages.data else ""

    if not text:
        return REFUSAL

    # If assistant explicitly returns refusal phrase, enforce it
    if REFUSAL in text:
        return REFUSAL

    # Gatekeeping:
    # - Normal factual answers must include at least one quote from docs.
    # - BUT evaluation mode is allowed without quotes (it should contain "Estimated Score:")
    is_evaluation = ("Estimated Score:" in text)

    if not is_evaluation:
        has_quote = ('"' in text) or ("“" in text) or ("”" in text)
        if not has_quote:
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
        "Ask any GS-1 History / Indian Art & Culture question.\n\n"
        "Answer Writing:\n"
        "Paste your answer and say: 'Evaluate my answer (10/15/20 marker)'.\n\n"
        "Faculty Upload:\n"
        "/uploaddoc <ADMIN_SECRET>\n"
        "Then send PDF/DOC."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()
    answer = await docs_only_answer(user_text)

    # Telegram safe length
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

        # Reset assistant cache so new docs are immediately available in retrieval behavior
        global _ASSISTANT_ID
        _ASSISTANT_ID = None

        await update.message.reply_text(f"✅ Uploaded & indexed.\nFile ID: {file_id}")

    except Exception as e:
        logger.exception(e)
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
