import os
import time
import logging

import ollama

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

TRANSLATOR_MODEL = "gemma4:e4b"
MEDICAL_MODEL = "medgemma1.5:4b"

# Keep models warm for 1 hour to prevent unloading delays between turns
KEEP_ALIVE = "1h"

# History turns to feed back into Brain 1 for context resolution
MAX_HISTORY_TURNS = 3

DEBUG_MODE = False

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

logger = logging.getLogger("cebuano_doctor")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# CUSTOM ERROR TYPE
# ---------------------------------------------------------------------------

class CebuanoDoctorError(Exception):
    def __init__(self, friendly_message_cebuano, technical_detail=""):
        super().__init__(technical_detail or friendly_message_cebuano)
        self.friendly_message_cebuano = friendly_message_cebuano
        self.technical_detail = technical_detail


# ---------------------------------------------------------------------------
# PROMPT LOADING
# ---------------------------------------------------------------------------

def _load_prompt(filename):
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


_CEBUANO_TO_ENGLISH_PROMPT = _load_prompt("cebuano_to_english.txt")
_MEDICAL_RESPONSE_PROMPT = _load_prompt("medical_response.txt")
_ENGLISH_TO_CEBUANO_PROMPT = _load_prompt("english_to_cebuano.txt")


def list_prompt_files():
    if not os.path.isdir(PROMPTS_DIR):
        return []
    return sorted(f for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt"))


def get_prompt_content(filename):
    path = os.path.join(PROMPTS_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise CebuanoDoctorError(
            "There are no prompts",
            f"Prompt file not found: {path}",
        )


def set_debug_mode(enabled: bool):
    global DEBUG_MODE
    DEBUG_MODE = enabled


def is_debug_mode() -> bool:
    return DEBUG_MODE


# ---------------------------------------------------------------------------
# ADAPTIVE GENERATION LENGTH
# ---------------------------------------------------------------------------

def _adaptive_num_predict(user_content, stage_label):
    word_count = max(1, len(user_content.split()))

    if stage_label in ("cebuano_to_english", "english_to_cebuano"):
        # Translation output is roughly proportional to input length.
        floor, ceiling, tokens_per_word = 60, 400, 3
    else:  # medical_response
        floor, ceiling, tokens_per_word = 220, 768, 10

    estimate = int(word_count * tokens_per_word)
    return max(floor, min(ceiling, estimate))


def _build_options(user_content, stage_label):
    """Shared options dict builder used by both the normal and streaming call paths."""
    is_translation = "translation" in stage_label or "cebuano" in stage_label
    return {
        "num_predict": _adaptive_num_predict(user_content, stage_label),
        # Low randomness for translation accuracy; slightly higher for
        # medical reasoning, which needs to weigh multiple possibilities.
        "temperature": 0.2 if is_translation else 0.3,
    }


def _map_ollama_exception(e, stage_label):
    if isinstance(e, ollama.ResponseError):
        status = getattr(e, "status_code", None)
        if status == 404 or "not found" in str(e).lower():
            return CebuanoDoctorError(
                "Pasensya, wala pa na-install ang kinahanglan nga AI model. "
                "Palihog i-check ang README kung giunsa pag-install sa mga modelo.",
                f"Ollama model not found for stage '{stage_label}': {e}",
            )
        return CebuanoDoctorError(
            "Pasensya, adunay problema sa pagproseso sa imong pangutana. "
            "Palihog sulayi pag-usab.",
            f"Ollama returned an error at stage '{stage_label}': {e}",
        )
    if isinstance(e, ConnectionError):
        return CebuanoDoctorError(
            "Pasensya, dili ma-abot ang lokal nga AI service. Siguroha nga "
            "nagdagan ang Ollama sa imong kompyuter, unya sulayi pag-usab.",
            f"Could not connect to local Ollama service at stage '{stage_label}': {e}",
        )
    if isinstance(e, TimeoutError):
        return CebuanoDoctorError(
            "Pasensya, dugay kaayo mo-response ang AI. Palihog sulayi pag-usab.",
            f"Timeout at stage '{stage_label}': {e}",
        )
    return CebuanoDoctorError(
        "Pasensya, adunay wala damha nga problema. Palihog sulayi pag-usab.",
        f"Unexpected error at stage '{stage_label}': {e}",
    )


# ---------------------------------------------------------------------------
# LOW-LEVEL OLLAMA CALLS (non-streaming and streaming)
# ---------------------------------------------------------------------------

def _call_ollama(model_name, system_prompt, user_content, stage_label):
    options = _build_options(user_content, stage_label)

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            keep_alive=KEEP_ALIVE,
            options=options,
            think=False,
        )
    except Exception as e:
        raise _map_ollama_exception(e, stage_label)

    try:
        text = response["message"]["content"]
    except (KeyError, TypeError) as e:
        raise CebuanoDoctorError(
            "Pasensya, wala kasabot ang AI sa imong pangutana. Palihog sulayi pag-usab.",
            f"Unexpected response shape at stage '{stage_label}': {e}",
        )

    text = (text or "").strip()
    if not text:
        raise CebuanoDoctorError(
            "Pasensya, walay na-generate nga tubag. Palihog sulayi pag-usab.",
            f"Empty response at stage '{stage_label}'.",
        )
    return text


def _call_ollama_stream(model_name, system_prompt, user_content, stage_label, on_chunk=None):
    options = _build_options(user_content, stage_label)

    accumulated = []
    try:
        stream = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            keep_alive=KEEP_ALIVE,
            options=options,
            stream=True,
            think=False,
        )
        for chunk in stream:
            try:
                piece = chunk["message"]["content"]
            except (KeyError, TypeError):
                continue
            if not piece:
                continue
            accumulated.append(piece)
            if on_chunk:
                on_chunk(piece)
    except Exception as e:
        raise _map_ollama_exception(e, stage_label)

    text = "".join(accumulated).strip()
    if not text:
        raise CebuanoDoctorError(
            "Pasensya, walay na-generate nga tubag. Palihog sulayi pag-usab.",
            f"Empty response at stage '{stage_label}'.",
        )
    return text



# ---------------------------------------------------------------------------
# MODEL WARMUP ROUTINE
# ---------------------------------------------------------------------------

def warmup_models():
    """Pre-loads models into memory in a background thread at startup."""
    try:
        logger.info("Warming up models...")
        ollama.chat(
            model=TRANSLATOR_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            keep_alive=KEEP_ALIVE,
            options={"num_predict": 1},
            think=False,
        )
        ollama.chat(
            model=MEDICAL_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            keep_alive=KEEP_ALIVE,
            options={"num_predict": 1},
            think=False,
        )
        logger.info("Models warm and ready!")
    except Exception as e:
        logger.warning("Warmup skipped or failed: %s", e)


# ---------------------------------------------------------------------------
# BRAIN STAGES & CONVERSATION HISTORY
# ---------------------------------------------------------------------------

def translate_cebuano_to_english(cebuano_text, history_context=""):
    user_message = cebuano_text
    if history_context:
        user_message = (
            f"[Previous context, for reference only -- do not translate this part]\n"
            f"{history_context}\n\n[Message to translate]\n{cebuano_text}"
        )
    return _call_ollama(
        TRANSLATOR_MODEL,
        _CEBUANO_TO_ENGLISH_PROMPT,
        user_message,
        stage_label="cebuano_to_english",
    )


def generate_medical_response(english_question):
    return _call_ollama(
        MEDICAL_MODEL,
        _MEDICAL_RESPONSE_PROMPT,
        english_question,
        stage_label="medical_response",
    )


def translate_english_to_cebuano(english_text):
    return _call_ollama(
        TRANSLATOR_MODEL,
        _ENGLISH_TO_CEBUANO_PROMPT,
        english_text,
        stage_label="english_to_cebuano",
    )


def translate_english_to_cebuano_stream(english_text, on_chunk=None):
    return _call_ollama_stream(
        TRANSLATOR_MODEL,
        _ENGLISH_TO_CEBUANO_PROMPT,
        english_text,
        stage_label="english_to_cebuano",
        on_chunk=on_chunk,
    )


class ConversationHistory:
    def __init__(self, max_turns=MAX_HISTORY_TURNS):
        self.max_turns = max_turns
        self._turns = []

    def add_turn(self, cebuano_user, english_user, cebuano_reply):
        self._turns.append(
            {
                "cebuano_user": cebuano_user,
                "english_user": english_user,
                "cebuano_reply": cebuano_reply,
            }
        )
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def english_context_block(self):
        if not self._turns:
            return ""
        lines = []
        for turn in self._turns:
            lines.append(f"Patient said: {turn['english_user']}")
        return "\n".join(lines)

    def clear(self):
        self._turns = []


# ---------------------------------------------------------------------------
# HIGH-LEVEL PIPELINE FUNCTION
# ---------------------------------------------------------------------------

def process_cebuano_message(cebuano_text, history: ConversationHistory, on_status=None, on_cebuano_chunk=None):

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

    # Stage 3: English -> Cebuano. Streamed when the caller wants live
    # updates (e.g. the UI filling in the reply bubble as it's generated);
    # otherwise the plain, wait-for-completion call.
    _status("Nag-andam sa tubag sa Cebuano...")
    t0 = time.time()
    if on_cebuano_chunk:
        cebuano_answer = translate_english_to_cebuano_stream(english_answer, on_cebuano_chunk)
    else:
        cebuano_answer = translate_english_to_cebuano(english_answer)
    timings["english_to_cebuano_sec"] = round(time.time() - t0, 2)

    timings["total_sec"] = round(
        timings["cebuano_to_english_sec"]
        + timings["medical_response_sec"]
        + timings["english_to_cebuano_sec"],
        2,
    )

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