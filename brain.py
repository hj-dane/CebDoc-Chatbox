"""
brain.py
--------
This file is the "brain" of Cebuano Doctor.

It is responsible for:
  1. Talking to the local Ollama server (never the internet).
  2. Loading the three internal system prompts from the prompts/ folder.
  3. Running the three-stage pipeline:
         Cebuano -> English -> Medical response -> Cebuano
  4. Keeping a short conversation history so follow-up questions make sense.
  5. Catching and translating technical errors into friendly Cebuano
     messages that main.py can show to the user.

main.py never talks to Ollama directly. It only calls the functions in
this file. That separation is what keeps the pipeline "hidden" from the
user interface -- main.py only ever sees a Cebuano-in, Cebuano-out
function call.
"""

import os
import time
import logging

import ollama

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Exact model names required by the assignment.
TRANSLATOR_MODEL = "gemma4:e4b"
MEDICAL_MODEL = "medgemma1.5:4b"

# How long (in seconds) Ollama should keep each model loaded in memory
# after a request. Keeping models "warm" avoids the multi-second reload
# penalty every time the pipeline runs (Brain 1 and Brain 3 both use the
# same translator model, so this especially helps them).
KEEP_ALIVE = "10m"

# How many previous user/assistant Cebuano turns to remember and feed
# back into the pipeline for context. Kept small on purpose -- large
# histories make every one of the three model calls slower.
MAX_HISTORY_TURNS = 3

# Toggle debug/evaluation mode here, or override at runtime with
# set_debug_mode(). Debug mode must be OFF by default for normal users.
DEBUG_MODE = False

# Folder that stores the three internal prompt files.
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

# Basic logger. In debug mode we print pipeline details; otherwise this
# stays quiet so the console looks clean for a normal user.
logger = logging.getLogger("cebuano_doctor")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# CUSTOM ERROR TYPE
# ---------------------------------------------------------------------------

class CebuanoDoctorError(Exception):
    """
    Raised whenever something goes wrong inside the pipeline.

    We always attach a friendly Cebuano message so main.py can show it
    directly to the user without needing to know *why* something failed.
    """

    def __init__(self, friendly_message_cebuano, technical_detail=""):
        super().__init__(technical_detail or friendly_message_cebuano)
        self.friendly_message_cebuano = friendly_message_cebuano
        self.technical_detail = technical_detail


# ---------------------------------------------------------------------------
# PROMPT LOADING (loaded once, reused for every request)
# ---------------------------------------------------------------------------

def _load_prompt(filename):
    """Read one of the three internal system prompt files from disk."""
    path = os.path.join(PROMPTS_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise CebuanoDoctorError(
            "Pasensya, adunay problema sa pag-set up sa chatbot. "
            "Palihog i-check ang mga prompt files.",
            f"Missing prompt file: {path}",
        )


# Load all three prompts once at import time instead of re-reading the
# files from disk on every single message.
_CEBUANO_TO_ENGLISH_PROMPT = _load_prompt("cebuano_to_english.txt")
_MEDICAL_RESPONSE_PROMPT = _load_prompt("medical_response.txt")
_ENGLISH_TO_CEBUANO_PROMPT = _load_prompt("english_to_cebuano.txt")


# ---------------------------------------------------------------------------
# PROMPT BROWSING (used by the debug panel in main.py)
# ---------------------------------------------------------------------------
#
# These two functions let the debug UI list the available internal prompt
# files and show the raw content of whichever one the developer picks,
# for evaluation purposes. They read straight from PROMPTS_DIR on disk,
# so they always reflect whatever .txt files actually exist there --
# they don't rely on the three prompts already loaded above.

def list_prompt_files():
    """
    Return a sorted list of prompt filenames currently found in the
    prompts/ folder (e.g. ["cebuano_to_english.txt", ...]).

    Returns an empty list if the folder is missing or has no .txt files
    in it -- callers should treat an empty list as "there are no prompts".
    """
    if not os.path.isdir(PROMPTS_DIR):
        return []
    return sorted(f for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt"))


def get_prompt_content(filename):
    """
    Return the raw text content of one prompt file by name (as returned
    by list_prompt_files()). Raises CebuanoDoctorError if it can't be
    read, so the UI can show a friendly message instead of crashing.
    """
    path = os.path.join(PROMPTS_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise CebuanoDoctorError(
            "There are no prompts",
            f"Prompt file not found: {path}",
        )


# ---------------------------------------------------------------------------
# DEBUG MODE CONTROL
# ---------------------------------------------------------------------------

def set_debug_mode(enabled: bool):
    """Turn developer/evaluation debug mode on or off at runtime."""
    global DEBUG_MODE
    DEBUG_MODE = enabled


def is_debug_mode() -> bool:
    return DEBUG_MODE


# ---------------------------------------------------------------------------
# LOW-LEVEL OLLAMA CALL (shared by all three brains)
# ---------------------------------------------------------------------------

def _call_ollama(model_name, system_prompt, user_content, stage_label):
    """
    Send one request to a local Ollama model and return the plain text
    reply. All three brains funnel through this function so error
    handling only has to be written once.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            keep_alive=KEEP_ALIVE,
        )
    except ollama.ResponseError as e:
        # Ollama is running, but it complained -- usually means the
        # requested model isn't pulled/installed yet.
        status = getattr(e, "status_code", None)
        if status == 404 or "not found" in str(e).lower():
            raise CebuanoDoctorError(
                "Pasensya, wala pa na-install ang kinahanglan nga AI model. "
                "Palihog i-check ang README kung giunsa pag-install sa mga "
                "modelo.",
                f"Ollama model not found for stage '{stage_label}': {e}",
            )
        raise CebuanoDoctorError(
            "Pasensya, adunay problema sa pagproseso sa imong pangutana. "
            "Palihog sulayi pag-usab.",
            f"Ollama returned an error at stage '{stage_label}': {e}",
        )
    except ConnectionError as e:
        # This is the classic "Ollama isn't running" case.
        raise CebuanoDoctorError(
            "Pasensya, dili ma-abot ang lokal nga AI service. Siguroha nga "
            "nagdagan ang Ollama sa imong kompyuter, unya sulayi pag-usab.",
            f"Could not connect to local Ollama service at stage "
            f"'{stage_label}': {e}",
        )
    except TimeoutError as e:
        raise CebuanoDoctorError(
            "Pasensya, dugay kaayo mo-response ang AI. Palihog sulayi pag-usab.",
            f"Timeout at stage '{stage_label}': {e}",
        )
    except Exception as e:
        # Catch-all so an unexpected library error never crashes the app
        # or shows a raw Python traceback to the user.
        raise CebuanoDoctorError(
            "Pasensya, adunay wala damha nga problema. Palihog sulayi pag-usab.",
            f"Unexpected error at stage '{stage_label}': {e}",
        )

    try:
        text = response["message"]["content"]
    except (KeyError, TypeError) as e:
        raise CebuanoDoctorError(
            "Pasensya, wala kasabot ang AI sa imong pangutana. Palihog "
            "sulayi pag-usab.",
            f"Unexpected response shape at stage '{stage_label}': {e}",
        )

    text = (text or "").strip()
    if not text:
        raise CebuanoDoctorError(
            "Pasensya, walay na-generate nga tubag. Palihog sulayi pag-usab.",
            f"Empty response at stage '{stage_label}'.",
        )
    return text


# ---------------------------------------------------------------------------
# BRAIN 1 -- CEBUANO -> ENGLISH
# ---------------------------------------------------------------------------

def translate_cebuano_to_english(cebuano_text, history_context=""):
    """
    Translate a Cebuano healthcare message into English.

    history_context (optional) is a short block of recent prior turns,
    already in English, so the translator can resolve follow-ups like
    "Sakit gihapon hangtod karon" without re-translating the whole
    conversation.
    """
    user_message = cebuano_text
    if history_context:
        user_message = (
            f"[Previous context, for reference only -- do not translate this "
            f"part]\n{history_context}\n\n[Message to translate]\n{cebuano_text}"
        )
    return _call_ollama(
        TRANSLATOR_MODEL,
        _CEBUANO_TO_ENGLISH_PROMPT,
        user_message,
        stage_label="cebuano_to_english",
    )


# ---------------------------------------------------------------------------
# BRAIN 2 -- MEDICAL REASONING
# ---------------------------------------------------------------------------

def generate_medical_response(english_question):
    """Send the English question to MedGemma and get back medical guidance."""
    return _call_ollama(
        MEDICAL_MODEL,
        _MEDICAL_RESPONSE_PROMPT,
        english_question,
        stage_label="medical_response",
    )


# ---------------------------------------------------------------------------
# BRAIN 3 -- ENGLISH -> CEBUANO
# ---------------------------------------------------------------------------

def translate_english_to_cebuano(english_text):
    """Translate MedGemma's English medical response back into Cebuano."""
    return _call_ollama(
        TRANSLATOR_MODEL,
        _ENGLISH_TO_CEBUANO_PROMPT,
        english_text,
        stage_label="english_to_cebuano",
    )


# ---------------------------------------------------------------------------
# CONVERSATION HISTORY
# ---------------------------------------------------------------------------

class ConversationHistory:
    """
    Keeps a small rolling log of the conversation so follow-up questions
    can be understood, without ever letting the log grow indefinitely.

    Each turn stores both the Cebuano text and its English translation,
    because Brain 1 needs English context (see translate_cebuano_to_english)
    while the UI needs the Cebuano text.
    """

    def __init__(self, max_turns=MAX_HISTORY_TURNS):
        self.max_turns = max_turns
        self._turns = []  # list of dicts: {cebuano_user, english_user, cebuano_reply}

    def add_turn(self, cebuano_user, english_user, cebuano_reply):
        self._turns.append(
            {
                "cebuano_user": cebuano_user,
                "english_user": english_user,
                "cebuano_reply": cebuano_reply,
            }
        )
        # Trim from the front so we never keep more than max_turns.
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def english_context_block(self):
        """
        Build a short plain-text block of prior English exchanges, used
        only to help Brain 1 resolve follow-up questions. Returns "" if
        there is no history yet.
        """
        if not self._turns:
            return ""
        lines = []
        for turn in self._turns:
            lines.append(f"Patient said: {turn['english_user']}")
        return "\n".join(lines)

    def clear(self):
        self._turns = []


# ---------------------------------------------------------------------------
# HIGH-LEVEL PIPELINE FUNCTION (this is what main.py calls)
# ---------------------------------------------------------------------------

def process_cebuano_message(cebuano_text, history: ConversationHistory, on_status=None):
    """
    Run the full three-stage pipeline for one user message.

    Parameters:
        cebuano_text : the raw Cebuano message typed by the user.
        history      : a ConversationHistory instance to read/update.
        on_status    : optional callback, called with a short Cebuano
                       status string (e.g. "Gisabtan ang imong pangutana...")
                       right before each stage starts, so the UI can show
                       a temporary status indicator without ever seeing
                       intermediate model output.

    Returns:
        A dict with at least {"reply": <final Cebuano text>}.
        If debug mode is on, also includes the intermediate values and
        per-stage timings, for evaluation purposes only.
    """
    cebuano_text = (cebuano_text or "").strip()
    if not cebuano_text:
        raise CebuanoDoctorError(
            "Palihog pagsulat og pangutana sa dili pa mo-send.",
            "Empty user message.",
        )

    def _status(msg):
        if on_status:
            on_status(msg)

    timings = {}
    context_block = history.english_context_block()

    # Stage 1: Cebuano -> English
    _status("Gisabtan ang imong pangutana...")
    t0 = time.time()
    english_question = translate_cebuano_to_english(cebuano_text, context_block)
    timings["cebuano_to_english_sec"] = round(time.time() - t0, 2)

    # Stage 2: Medical reasoning
    _status("Nag-andam sa medikal nga tubag...")
    t0 = time.time()
    english_answer = generate_medical_response(english_question)
    timings["medical_response_sec"] = round(time.time() - t0, 2)

    # Stage 3: English -> Cebuano
    _status("Nag-andam sa tubag sa Cebuano...")
    t0 = time.time()
    cebuano_answer = translate_english_to_cebuano(english_answer)
    timings["english_to_cebuano_sec"] = round(time.time() - t0, 2)

    timings["total_sec"] = round(
        timings["cebuano_to_english_sec"]
        + timings["medical_response_sec"]
        + timings["english_to_cebuano_sec"],
        2,
    )

    # Update history with the English translation of the user's question
    # (used for future follow-up context) and the final Cebuano reply.
    history.add_turn(cebuano_text, english_question, cebuano_answer)

    result = {"reply": cebuano_answer}

    if DEBUG_MODE:
        result["debug"] = {
            "cebuano_input": cebuano_text,
            "english_translation": english_question,
            "english_medical_response": english_answer,
            "final_cebuano_response": cebuano_answer,
            "timings": timings,
        }
        logger.info("---- DEBUG: pipeline details ----")
        for key, value in result["debug"].items():
            logger.info("%s: %s", key, value)
        logger.info("----------------------------------")

    return result