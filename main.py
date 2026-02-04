import os
import re
import logging
import tempfile
import asyncio
from typing import List, Optional, Tuple

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

# --- PDF/OCR ---
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

D) What is MISSING / WEAK (Gaps)

E) What is BAD (Critical faults, if any)

F) UPSC-Level Improvements (Actionable Tips)

G) Value Addition (Rewrite Add-on)

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

# ---------------- Evaluation-only system prompt (NO DOCS; UPSC Rubric Only) ----------------
EVAL_SYSTEM_PROMPT = """
You are a strict UPSC GS-1 examiner and mentor.

You ONLY evaluate answers that belong to GS-1 History & Indian Art and Culture
(Ancient/Medieval/Modern Indian History, prescribed World History themes, Post-Independence as historical processes, Indian Art & Culture).
If the answer is clearly not from GS-1 History/Art & Culture, refuse:

"Refusal (Out of GS-1 History/Art & Culture): I can’t evaluate this because it is not GS-1 History/Art & Culture."

Evaluation format (MANDATORY):
A) Overall Examiner Impression (1–2 lines)
B) Estimated Score: X / MAX
   - Add: "This is an estimated, approximate score — not an official UPSC marking."
C) What is GOOD (Strengths) (bullets)
D) What is MISSING / WEAK (Gaps) (bullets)
E) What is BAD (Critical faults, if any) (bullets)
F) UPSC-Level Improvements (Actionable tips) (bullets)
G) Value Addition (Rewrite Add-on)
   - Provide a rewritten improved answer in UPSC style within the expected word limit for the marker.

Marker mapping:
- 10 marker ≈ 150 words
- 15 marker ≈ 250 words
- 20 marker ≈ 350–400 words

Tone: strict, examiner-like. No fluff.
""".strip()


# ---------------- Utilities: query wrapping + detectors ----------------
def wrap_user_query(user_text: str) -> str:
    """
    Hard-pin output format every time (prevents 'simple paragraph' answers).
    This is sent as the USER message content into the assistant thread.
    """
    return f"""
Follow these operating rules STRICTLY:
1) Only GS-1 History + Indian Art & Culture.
2) Use ONLY faculty documents via file_search. No outside knowledge.
3) If not supported by faculty documents, reply EXACTLY: {REFUSAL}
4) For factual/content answers, include at least ONE short direct quote (1–2 lines) from the faculty documents.
5) If question asks 150/250 words or is MAINS-style → MUST output:
   - Introduction
   - Body (analytical subheadings)
   - Conclusion
   - Chronological Timeline (5–10 bullets)
   - Conceptual Mindmap (ASCII)
   - High-Value Keywords (8–15)
   - PYQ Frequency Band (High/Medium/Low) without inventing years
6) If user asks MCQs → follow UPSC Prelims format with Answer Key + elimination for each wrong option.

User question:
{user_text}
""".strip()


def is_mcq_request(user_text: str) -> bool:
    t = user_text.lower()
    return any(k in t for k in ["mcq", "mcqs", "multiple choice", "prelims", "objective", "choose the correct"])


def is_mains_request(user_text: str) -> bool:
    t = user_text.lower()
    if re.search(r"\b(150|250)\b", t):
        return True
    if "mains" in t:
        return True
    # Common mains directives
    return any(k in t for k in ["discuss", "analyse", "analyze", "critically", "examine", "comment", "elucidate", "explain"])


def looks_like_mains_pack(text: str) -> bool:
    t = text.lower()
    must = ["introduction", "conclusion", "timeline", "mindmap", "keywords", "pyq frequency"]
    return all(m in t for m in must)


def split_telegram_chunks(text: str, limit: int = 3900) -> List[str]:
    """
    Telegram hard limit ~4096. Use 3900 to be safe.
    Splits on paragraph boundaries where possible.
    """
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


# ---------------- Admin check ----------------
def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_TELEGRAM_IDS


# ---------------- Upload helpers (Faculty docs) ----------------
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
    if hasattr(client, "vector_stores"):
        client.vector_stores.files.create(vector_store_id=VECTOR_STORE_ID, file_id=file_id)
        return

    if hasattr(client, "beta") and hasattr(client.beta, "vector_stores"):
        client.beta.vector_stores.files.create(vector_store_id=VECTOR_STORE_ID, file_id=file_id)
        return

    raise RuntimeError(
        "Your OpenAI SDK does not support vector store attach (no vector_stores). "
        "Clear build cache deploy with openai==1.56.0 OR attach manually in OpenAI Dashboard."
    )


def upload_and_index_doc(local_path: str) -> str:
    file_id = upload_file_to_openai(local_path)
    attach_file_to_vector_store(file_id)
    return file_id


# ---------------- Docs-only Answer (Assistants API) ----------------
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
    text = messages.data[0].content[0].text.value.strip() if messages.data else ""
    return text


def docs_only_answer_sync(user_text: str) -> str:
    wrapped = wrap_user_query(user_text)

    text = _assistant_run(wrapped)
    logger.info("RAW_OUTPUT_HEAD=%s", (text[:600] if text else "").replace("\n", "\\n"))

    if not text:
        return REFUSAL

    if REFUSAL in text:
        return REFUSAL

    # Gatekeeping for docs-only outputs:
    # Must contain a quote to prove retrieval happened (unless evaluation mode, but eval is handled separately now).
    has_quote = ('"' in text) or ("“" in text) or ("”" in text)
    if not has_quote:
        return REFUSAL

    # UPSC MAINS enforcement second pass (only for mains-type, not MCQs)
    if (not is_mcq_request(user_text)) and is_mains_request(user_text):
        if not looks_like_mains_pack(text):
            reform_request = f"""
Reformat the following answer into the mandated UPSC MAINS ENRICHMENT PACK structure:
- Introduction
- Body (analytical subheadings)
- Conclusion
- Chronological Timeline (5–10 bullets)
- Conceptual Mindmap (ASCII)
- High-Value Keywords (8–15)
- PYQ Frequency Band (High/Medium/Low) without inventing years

CRITICAL CONSTRAINTS:
- Do NOT add any new facts beyond what is already present OR what is retrieved from faculty documents.
- Use ONLY faculty documents via file_search.
- Include at least ONE short direct quote (1–2 lines) as evidence.
- If not supported, reply EXACTLY: {REFUSAL}

TEXT TO REFORMAT:
{text}
""".strip()

            text2 = _assistant_run(reform_request)
            logger.info("REFORMAT_OUTPUT_HEAD=%s", (text2[:600] if text2 else "").replace("\n", "\\n"))

            if text2 and (REFUSAL not in text2):
                has_quote2 = ('"' in text2) or ("“" in text2) or ("”" in text2)
                if has_quote2:
                    text = text2
                else:
                    return REFUSAL
            elif text2 and (REFUSAL in text2):
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


# ---------------- Evaluation detection + marker parsing ----------------
EVAL_PAT = re.compile(r"\bevaluate\s+my\s+(answer|pdf)\b", re.IGNORECASE)


def is_evaluation_request(text: str) -> bool:
    return bool(text and EVAL_PAT.search(text))


def parse_marker_max(text: str, answer_text: str) -> int:
    """
    Decide MAX = 10/15/20 from user text or inferred word count.
    """
    t = (text or "").lower()
    if "20" in t and "marker" in t:
        return 20
    if "15" in t and "marker" in t:
        return 15
    if "10" in t and "marker" in t:
        return 10

    # Infer from word count
    wc = len((answer_text or "").split())
    if wc >= 330:
        return 20
    if wc >= 200:
        return 15
    return 10


def strip_eval_directive(text: str) -> str:
    """
    If a student pastes answer + writes 'Evaluate my answer ...' in same message,
    remove the directive lines to keep only the answer body.
    """
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines()]
    kept = []
    for ln in lines:
        if EVAL_PAT.search(ln):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


# ---------------- OCR / PDF extraction ----------------
def _clean_extracted_text(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_text_from_pdf(path: str, max_pages: int = 25) -> Tuple[str, str]:
    """
    Returns (text, mode) where mode in {"typed_pdf", "ocr_pdf"}.
    """
    # 1) Try typed PDF extraction
    try:
        chunks = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                t = page.extract_text() or ""
                if t.strip():
                    chunks.append(t)
        typed_text = _clean_extracted_text("\n\n".join(chunks))
    except Exception:
        typed_text = ""

    # If enough text, treat as typed PDF
    if len(typed_text.split()) >= 40:
        return typed_text, "typed_pdf"

    # 2) OCR for scanned PDFs
    try:
        images = convert_from_path(path, dpi=250, first_page=1, last_page=max_pages)
        ocr_parts = []
        for img in images:
            # mild help for OCR
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


# ---------------- UPSC evaluation (Chat Completions, NO file_search) ----------------
def evaluate_answer_sync(answer_text: str, marker_max: int) -> str:
    """
    UPSC-style evaluation, independent of faculty docs.
    """
    answer_text = (answer_text or "").strip()
    if not answer_text:
        return "⚠️ I couldn’t read any text from your answer. Please upload a clearer scan or paste typed text."

    user_prompt = f"""
Marker: {marker_max} marker (MAX = {marker_max})
Expected length guidance:
- 10 marker ≈ 150 words
- 15 marker ≈ 250 words
- 20 marker ≈ 350–400 words

Now evaluate the student's answer below:

--- STUDENT ANSWER START ---
{answer_text}
--- STUDENT ANSWER END ---
""".strip()

    # Retry loop for transient issues
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
        "1) Paste your answer OR upload PDF / photo.\n"
        "2) Then send: 'Evaluate my answer (10/15/20 marker)'.\n\n"
        "Faculty Upload:\n"
        "/uploaddoc <ADMIN_SECRET>\n"
        "Then send PDF/DOC."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()

    # ---- Evaluation request (text-based) ----
    if is_evaluation_request(user_text):
        # Case 1: Answer + directive in same message
        possible_answer = strip_eval_directive(user_text)
        if len(possible_answer.split()) >= 30:
            marker_max = parse_marker_max(user_text, possible_answer)
            out = await evaluate_answer(possible_answer, marker_max)
            for part in split_telegram_chunks(out, limit=3900):
                await update.message.reply_text(part)
            return

        # Case 2: Evaluate the last uploaded/pasted answer stored in user_data
        cached = context.user_data.get("last_answer_text", "")
        if not cached or len(cached.split()) < 10:
            await update.message.reply_text(
                "⚠️ I don’t have your answer text yet.\n"
                "Please paste your answer OR upload a PDF/photo first, then send: Evaluate my answer (10/15/20 marker)."
            )
            return

        marker_max = parse_marker_max(user_text, cached)
        out = await evaluate_answer(cached, marker_max)
        for part in split_telegram_chunks(out, limit=3900):
            await update.message.reply_text(part)
        return

    # ---- Normal Q&A (docs-only) ----
    answer = await docs_only_answer(user_text)
    for part in split_telegram_chunks(answer, limit=3900):
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
    await update.message.reply_text("✅ Now send the faculty PDF/DOC file to upload & index.")


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

        # ---------- Faculty upload flow ----------
        if context.user_data.get("awaiting_doc_upload"):
            if not ADMIN_SECRET or not ADMIN_TELEGRAM_IDS:
                await update.message.reply_text(
                    "❌ Upload system not configured.\n"
                    "Set ADMIN_SECRET and ADMIN_TELEGRAM_IDS in Render env vars."
                )
                return

            if not is_admin(user_id):
                await update.message.reply_text("❌ Only faculty can upload.")
                return

            await update.message.reply_text("⏳ Uploading & indexing faculty document...")

            file_id = await asyncio.to_thread(upload_and_index_doc, tmp_path)

            # Reset assistant cache so new docs are immediately available
            global _ASSISTANT_ID
            _ASSISTANT_ID = None

            await update.message.reply_text(f"✅ Uploaded & indexed.\nFile ID: {file_id}")
            return

        # ---------- Student submission flow ----------
        filename = (doc.file_name or "").lower()

        # PDF: extract typed text, else OCR
        if filename.endswith(".pdf") or (doc.mime_type == "application/pdf"):
            await update.message.reply_text("⏳ Reading your PDF (text/OCR)...")
            text, mode = await asyncio.to_thread(extract_text_from_pdf, tmp_path)
            context.user_data["last_answer_text"] = text

            if not text or len(text.split()) < 10:
                await update.message.reply_text(
                    "⚠️ I couldn’t read enough text from this PDF.\n"
                    "Tips: upload a clearer scan (straight, good light) OR paste typed text."
                )
                return

            # If caption asks for evaluation, evaluate immediately
            if is_evaluation_request(caption):
                marker_max = parse_marker_max(caption, text)
                out = await evaluate_answer(text, marker_max)
                for part in split_telegram_chunks(out, limit=3900):
                    await update.message.reply_text(part)
            else:
                await update.message.reply_text(
                    "✅ PDF received.\nNow send: Evaluate my answer (10/15/20 marker)."
                )
            return

        # TXT: read directly
        if filename.endswith(".txt") or (doc.mime_type == "text/plain"):
            await update.message.reply_text("⏳ Reading your text file...")
            with open(tmp_path, "rb") as f:
                raw = f.read()
            text = raw.decode("utf-8", errors="ignore")
            text = _clean_extracted_text(text)
            context.user_data["last_answer_text"] = text

            if is_evaluation_request(caption):
                marker_max = parse_marker_max(caption, text)
                out = await evaluate_answer(text, marker_max)
                for part in split_telegram_chunks(out, limit=3900):
                    await update.message.reply_text(part)
            else:
                await update.message.reply_text("✅ Answer received.\nNow send: Evaluate my answer (10/15/20 marker).")
            return

        # Unsupported docs for student evaluation (DOC/DOCX)
        await update.message.reply_text(
            "⚠️ For student answer evaluation, please upload:\n"
            "- PDF (typed or scanned)\n"
            "- Photo of written answer\n"
            "- Or paste the answer as text\n\n"
            "DOC/DOCX evaluation is not enabled yet."
        )

    except Exception as e:
        logger.exception(e)
        await update.message.reply_text("❌ File handling failed. Please try again.")

    finally:
        # Reset faculty upload flag if set
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
        # Get highest resolution photo
        photo = update.message.photo[-1]
        file_obj = await context.bot.get_file(photo.file_id)

        with tempfile.NamedTemporaryFile(delete=False, suffix="_photo.jpg") as tmp:
            tmp_path = tmp.name

        await file_obj.download_to_drive(custom_path=tmp_path)

        await update.message.reply_text("⏳ Reading your photo (OCR)...")
        text = await asyncio.to_thread(ocr_image_path, tmp_path)
        context.user_data["last_answer_text"] = text

        if not text or len(text.split()) < 10:
            await update.message.reply_text(
                "⚠️ I couldn’t read enough text from this image.\n"
                "Tips: use good light, keep page straight, dark pen, no shadows, zoom so text is clear."
            )
            return

        if is_evaluation_request(caption):
            marker_max = parse_marker_max(caption, text)
            out = await evaluate_answer(text, marker_max)
            for part in split_telegram_chunks(out, limit=3900):
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

    # Order matters: photos/docs before text
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.add_error_handler(error_handler)

    logger.info("✅ PastPulse bot running (polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
