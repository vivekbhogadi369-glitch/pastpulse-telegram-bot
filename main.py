import os
import re
import logging
import tempfile
import asyncio
from typing import List, Tuple

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

# ---- PDF/OCR deps ----
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract


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

# This must match EXACTLY what you want users to see when content is missing.
REFUSAL = "Not found in faculty documents. Ask faculty to upload the relevant material."

# ---------------- MERGED SYSTEM PROMPT (UPSC + GS-1 Boundary + Docs-Only) ----------------
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
- Ancient/Medieval/Modern Indian History
- Prescribed World History themes
- Post-Independence history ONLY as historical processes (no current affairs)
- Indian Art & Culture

========================
2) HARD REFUSAL BOUNDARY (NON-NEGOTIABLE)
========================
Do NOT answer:
- Polity/Constitution/Governance, Economy, Social Justice, IR/Current Affairs, Geography/Environment, S&T, Ethics/Internal Security/Essay

Mixed questions:
- Answer ONLY the historical/cultural part and refuse the rest in one line.

IMPORTANT: If the historical/cultural part is not supported by faculty documents, respond EXACTLY with:
{REFUSAL}

========================
3) PRELIMS MCQs (UPSC STANDARD — STRICT)
========================
If the user asks for MCQs:
- Statement-based, elimination-driven UPSC standard.
- Provide Questions first, then Answer Key with elimination for each wrong option.
- If unsupported by docs, reply EXACTLY: {REFUSAL}
- Include at least ONE short quote from documents.

========================
4) MAINS ANSWERS (MANDATORY ENRICHMENT PACK)
========================
If MAINS-style or 150/250 words:
- Introduction, Body (subheadings), Conclusion
- Timeline, ASCII mindmap, Keywords, PYQ Frequency Band (no invented years)
- If unsupported by docs, reply EXACTLY: {REFUSAL}
- Include at least ONE short quote from documents.

========================
5) QUALITY CONTROL
========================
- Examiner tone, no fluff.
- When in doubt, exclude content rather than guessing.
""".strip()

# ---------------- Evaluation-only system prompt (NO DOCS) ----------------
# FIXED: UPSC-style marking with minimum attempt marks and 0 only in true zero-cases.
EVAL_SYSTEM_PROMPT = """
You are a strict UPSC GS-1 examiner and mentor.

You ONLY evaluate answers that belong to GS-1 History & Indian Art and Culture
(Ancient/Medieval/Modern Indian History, prescribed World History themes, Post-Independence as historical processes, Indian Art & Culture).
If the answer is clearly not from GS-1 History/Art & Culture, refuse EXACTLY:

"Refusal (Out of GS-1 History/Art & Culture): I can’t evaluate this because it is not GS-1 History/Art & Culture."

========================
UPSC MARKING PRINCIPLE (NON-NEGOTIABLE)
========================
- DO NOT default to 0 for weak answers.
- Give minimum marks if there is a genuine attempt (even if poor).
- "Estimated Score: 0 / MAX" is allowed ONLY if:
  (a) the answer is blank/near-blank, OR
  (b) completely unrelated to GS-1 History/Art & Culture, OR
  (c) text is meaningless/gibberish with no interpretable content.

If there is any interpretable attempt on-topic, award at least:
- 10 marker: 1 to 2 marks
- 15 marker: 1 to 3 marks
- 20 marker: 2 to 4 marks

Score dimension-wise (UPSC style) and then total:
1) Structure (Intro-Body-Conclusion, flow)
2) Relevance (addresses directive + demand)
3) Content (facts/examples/terms)
4) Analysis (causation, significance, continuity-change, balance)
5) Presentation (clarity, keywords, subheadings)

========================
MANDATORY OUTPUT FORMAT
========================
A) Overall Examiner Impression (1–2 lines)

B) Estimated Score: X / MAX
   - Add: "This is an estimated, approximate score — not an official UPSC marking."
   - X must be realistic given the attempt.

C) Marking Breakdown (dimension-wise)
   - Structure: a/b
   - Relevance: a/b
   - Content: a/b
   - Analysis: a/b
   - Presentation: a/b
   (Choose b values consistent with MAX; keep it simple and coherent.)

D) What is GOOD (Strengths) (bullets)

E) What is MISSING / WEAK (Gaps) (bullets)

F) What is BAD (Critical faults, if any) (bullets)

G) UPSC-Level Improvements (Actionable tips) (bullets)

H) Value Addition (Rewrite Add-on)
   - Provide an improved rewritten answer in UPSC style within the expected word limit for the marker.
   - Keep it syllabus-bound; do not add irrelevant content.

Marker mapping:
- 10 marker ≈ 150 words
- 15 marker ≈ 250 words
- 20 marker ≈ 350–400 words

Tone: strict, examiner-like. No fluff.
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


def _clean_extracted_text(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_TELEGRAM_IDS


# ---------------- Evaluation detection ----------------
EVAL_PAT = re.compile(r"\bevaluate\s*my\s*answer\b", re.IGNORECASE)


def is_evaluation_request(text: str) -> bool:
    return bool(text and EVAL_PAT.search(text))


def parse_marker_max(text: str, answer_text: str) -> int:
    t = (text or "").lower()
    if "20" in t and "marker" in t:
        return 20
    if "15" in t and "marker" in t:
        return 15
    if "10" in t and "marker" in t:
        return 10

    wc = len((answer_text or "").split())
    if wc >= 330:
        return 20
    if wc >= 200:
        return 15
    return 10


def strip_eval_directive(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines()]
    kept = []
    for ln in lines:
        if is_evaluation_request(ln):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def looks_like_gibberish(answer_text: str) -> bool:
    """
    Heuristic to catch OCR garbage so we DON'T produce unfair 0 scores.
    If text is not readable, we ask for a clearer scan instead of evaluating.
    """
    t = (answer_text or "").strip()
    if not t:
        return True

    words = t.split()
    if len(words) < 10:
        return True

    # Words that contain at least 2 letters (basic "readability")
    readable_words = 0
    for w in words:
        if re.search(r"[A-Za-z].*[A-Za-z]", w):
            readable_words += 1

    readable_ratio = readable_words / max(1, len(words))
    # Too many symbols/noise
    non_alnum_ratio = len(re.findall(r"[^A-Za-z0-9\s]", t)) / max(1, len(t))

    # If most words are not readable OR text is heavily symbol-noisy -> gibberish
    if readable_ratio < 0.45:
        return True
    if non_alnum_ratio > 0.25:
        return True

    return False


# ---------------- OCR / PDF extraction ----------------
def extract_text_from_pdf(path: str, max_pages: int = 25) -> Tuple[str, str]:
    # 1) typed text
    typed = ""
    try:
        chunks = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:max_pages]:
                t = page.extract_text() or ""
                if t.strip():
                    chunks.append(t)
        typed = _clean_extracted_text("\n\n".join(chunks))
    except Exception:
        typed = ""

    if len(typed.split()) >= 40:
        return typed, "typed_pdf"

    # 2) OCR scanned
    try:
        images = convert_from_path(path, dpi=250, first_page=1, last_page=max_pages)
        ocr_parts = []
        for img in images:
            gray = img.convert("L")
            ocr_parts.append(pytesseract.image_to_string(gray))
        ocr_text = _clean_extracted_text("\n\n".join(ocr_parts))
        return ocr_text, "ocr_pdf"
    except Exception as e:
        logger.exception("PDF OCR failed: %s", e)
        return "", "ocr_pdf"


def ocr_image_path(path: str) -> str:
    try:
        img = Image.open(path)
        gray = img.convert("L")
        txt = pytesseract.image_to_string(gray)
        return _clean_extracted_text(txt)
    except Exception as e:
        logger.exception("Image OCR failed: %s", e)
        return ""


# ---------------- Faculty upload helpers ----------------
def upload_file_to_openai(local_path: str) -> str:
    with open(local_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="assistants")
    return uploaded.id


def attach_file_to_vector_store(file_id: str) -> None:
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
    return f"""
Follow these operating rules STRICTLY:
1) Only GS-1 History + Indian Art & Culture.
2) Use ONLY faculty documents via file_search. No outside knowledge.
3) If not supported by faculty documents, reply EXACTLY: {REFUSAL}
4) For factual/content answers, include at least ONE short direct quote (1–2 lines) from the faculty documents.

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

    # Must contain quote to prove retrieval
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


# ---------------- UPSC evaluation (Chat Completions, NO file_search) ----------------
def evaluate_answer_sync(answer_text: str, marker_max: int) -> str:
    answer_text = (answer_text or "").strip()

    if not answer_text or len(answer_text.split()) < 10:
        return "⚠️ I couldn’t read enough text from your answer. Upload a clearer scan/photo OR paste typed text."

    # NEW: If OCR is garbage, don't unfairly score 0 — ask for clearer input.
    if looks_like_gibberish(answer_text):
        return (
            "⚠️ The extracted text looks unclear/garbled (OCR noise), so a fair UPSC-style evaluation isn’t possible.\n"
            "Please upload a clearer scan/photo (straight page, good light, dark pen, no shadows) OR paste typed text."
        )

    user_prompt = f"""
Marker: {marker_max} marker (MAX = {marker_max})

Now evaluate the student's answer below. Apply UPSC marking principle: do NOT give 0 if there is a genuine attempt.

--- STUDENT ANSWER START ---
{answer_text}
--- STUDENT ANSWER END ---
""".strip()

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except APIConnectionError as e:
            logger.warning("Eval connection error attempt %s: %s", attempt + 1, e)
            if attempt < 2:
                continue
            return "⚠️ OpenAI connection error. Try again."
        except RateLimitError:
            if attempt < 2:
                continue
            return "⚠️ Too many requests. Try again after 1 minute."
        except APIStatusError as e:
            return f"⚠️ OpenAI API error: {e}"
        except Exception as e:
            logger.exception("Eval error: %s", e)
            return "Unexpected server error during evaluation. Please try again."


async def evaluate_answer(answer_text: str, marker_max: int) -> str:
    return await asyncio.to_thread(evaluate_answer_sync, answer_text, marker_max)


# ---------------- Telegram Handlers ----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ PastPulse AI is Live!\n"
        "Ask any GS-1 History / Indian Art & Culture question.\n\n"
        "Answer Writing:\n"
        "1) Upload photo/PDF OR paste your answer.\n"
        "2) Then send: Evaluate my answer (10/15/20 marker)\n\n"
        "Faculty Upload:\n"
        "/uploaddoc <ADMIN_SECRET>\n"
        "Then send PDF/DOC."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    raw_text = update.message.text.strip()

    # ---- Evaluation request ----
    if is_evaluation_request(raw_text):
        # If answer pasted + directive in same message
        maybe_answer = strip_eval_directive(raw_text)
        if len(maybe_answer.split()) >= 30:
            marker_max = parse_marker_max(raw_text, maybe_answer)
            out = await evaluate_answer(maybe_answer, marker_max)
            for part in split_telegram_chunks(out):
                await update.message.reply_text(part)
            return

        # else use last uploaded/pasted answer
        cached = context.user_data.get("last_answer_text", "")
        if not cached or len(cached.split()) < 10:
            await update.message.reply_text(
                "⚠️ I don’t have your answer text yet.\n"
                "Please upload a photo/PDF first OR paste your answer, then send: Evaluate my answer (10/15/20 marker)."
            )
            return

        marker_max = parse_marker_max(raw_text, cached)
        out = await evaluate_answer(cached, marker_max)
        for part in split_telegram_chunks(out):
            await update.message.reply_text(part)
        return

    # If user pasted a long answer (without saying evaluate), cache it for later evaluation
    # (optional quality-of-life; doesn't change behavior for normal questions)
    if len(raw_text.split()) >= 80 and not raw_text.lower().startswith("/"):
        context.user_data["last_answer_text"] = raw_text

    # ---- Normal docs-only Q&A ----
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
    caption = (update.message.caption or "").strip()

    tmp_path = None
    try:
        file_obj = await context.bot.get_file(doc.file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{doc.file_name or 'upload'}") as tmp:
            tmp_path = tmp.name
        await file_obj.download_to_drive(custom_path=tmp_path)

        # -------- Faculty upload flow --------
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

        # -------- Student PDF submission --------
        filename = (doc.file_name or "").lower()
        is_pdf = filename.endswith(".pdf") or (doc.mime_type == "application/pdf")

        if is_pdf:
            await update.message.reply_text("⏳ Reading your PDF (text/OCR)...")
            text, _mode = await asyncio.to_thread(extract_text_from_pdf, tmp_path)
            context.user_data["last_answer_text"] = text

            if not text or len(text.split()) < 10:
                await update.message.reply_text(
                    "⚠️ I couldn’t read enough text from this PDF.\n"
                    "Tips: upload a clearer scan (straight, good light) OR paste typed text."
                )
                return

            if is_evaluation_request(caption):
                marker_max = parse_marker_max(caption, text)
                out = await evaluate_answer(text, marker_max)
                for part in split_telegram_chunks(out):
                    await update.message.reply_text(part)
            else:
                await update.message.reply_text("✅ PDF received.\nNow send: Evaluate my answer (10/15/20 marker).")
            return

        await update.message.reply_text(
            "⚠️ For student evaluation, upload a PDF (typed/scanned) or a photo of the written answer.\n"
            "DOC/DOCX evaluation is not enabled yet."
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


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return

    caption = (update.message.caption or "").strip()

    tmp_path = None
    try:
        photo = update.message.photo[-1]  # highest resolution
        file_obj = await context.bot.get_file(photo.file_id)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_path = tmp.name

        await file_obj.download_to_drive(custom_path=tmp_path)

        await update.message.reply_text("⏳ Reading your photo (OCR)...")
        text = await asyncio.to_thread(ocr_image_path, tmp_path)

        context.user_data["last_answer_text"] = text

        if not text or len(text.split()) < 10:
            await update.message.reply_text(
                "⚠️ I couldn’t read enough text from this image.\n"
                "Tips: good light, page straight, dark pen, avoid shadows, zoom so text is clear."
            )
            return

        # If caption contains eval request, evaluate immediately
        if is_evaluation_request(caption):
            marker_max = parse_marker_max(caption, text)
            out = await evaluate_answer(text, marker_max)
            for part in split_telegram_chunks(out):
                await update.message.reply_text(part)
        else:
            await update.message.reply_text("✅ Image received.\nNow send: Evaluate my answer (10/15/20 marker).")

    except Exception as e:
        logger.exception(e)
        await update.message.reply_text("❌ Photo handling failed. Please try again.")
    finally:
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

    # IMPORTANT: photo/doc handlers must be before text handler
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.add_error_handler(error_handler)

    logger.info("✅ PastPulse bot running (polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
