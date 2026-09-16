# Cebuano Doctor

A local, offline Cebuano-language healthcare chatbot. All AI inference
happens on your own laptop through [Ollama](https://ollama.com) — no
cloud APIs, no internet-dependent translation service.

You type a healthcare question in Cebuano. Internally, three model
calls run in sequence to produce a Cebuano answer, but you only ever
see Cebuano in and Cebuano out.

```
Cebuano question  →  Gemma 4 (translate to English)
                  →  MedGemma (medical reasoning, in English)
                  →  Gemma 4 (translate back to Cebuano)
                  →  Cebuano answer
```

---

## 1. Installation

### 1.1 Install Ollama

Download and install Ollama for your OS from https://ollama.com/download,
then confirm it works:

```bash
ollama --version
```

### 1.2 Pull the required models

```bash
ollama pull gemma4:e4b
ollama pull medgemma1.5:4b
```

> These are the exact model names the application expects
> (`gemma4:e4b` and `medgemma1.5:4b`). If your local Ollama library uses
> different tag names, update `TRANSLATOR_MODEL` and `MEDICAL_MODEL` at
> the top of `brain.py` to match.

### 1.3 Install Python dependencies

Use Python 3.9+.

```bash
cd cebuano-doctor
pip install -r requirements.txt
```

`tkinter` (used for the chat window) ships with most standard Python
installations already, so nothing else is needed for the UI.

---

## 2. Running the application

Make sure Ollama is running in the background (on most systems it
starts automatically after cinstallation; otherwise run `ollama serve`
in a terminal), then:

```bash
python main.py
```

A chat window titled "Cebuano Doctor" will open. Type a Cebuano
healthcare question and press **Ipadala** (Send) or hit Enter.

---

## 3. Project structure

```
cebuano-doctor/
│
├── main.py                     # Chat window UI (Tkinter)
├── brain.py                    # Ollama pipeline, history, error handling
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
└── prompts/
    ├── cebuano_to_english.txt  # Brain 1 system prompt
    ├── medical_response.txt    # Brain 2 system prompt
    └── english_to_cebuano.txt  # Brain 3 system prompt
```

### `main.py`
Starts the Tkinter chat window, accepts Cebuano input, and shows the
Cebuano reply plus a temporary status line while processing. It calls
only `brain.process_cebuano_message(...)` — it has no knowledge of
Ollama, model names, or the internal prompts. Each pipeline call runs
on a background thread (`threading.Thread`) so the window never
freezes while the three models are working.

### `brain.py`
The pipeline engine. It:
- Loads the three prompt files from `prompts/` once, at startup.
- Talks to Ollama via the `ollama` Python library.
- Runs the three stages in order and returns only the final Cebuano
  text to `main.py` (plus optional debug data — see below).
- Keeps a small rolling conversation history so follow-up messages
  make sense.
- Converts any Ollama/connection/timeout error into a friendly Cebuano
  message, so the UI never has to show a Python traceback.

### `prompts/`
Plain text files holding the three internal system prompts. They are
part of the application, not something the end user ever types. Keeping
them as separate files (rather than hard-coded strings) makes them easy
to read, tweak, and reload without touching the Python code.

---

## 4. How the three internal "brains" work

Each brain is a separate call to a local Ollama model, using a
**system** message (the fixed instructions below) and a **user**
message (the content being processed at that stage). Keeping system
and user messages separate — rather than mashing them into one string
— gives the models a clearer, more reliable instruction boundary.

**Brain 1 — Cebuano → English** (`gemma4:e4b`)
Translates the user's Cebuano message into English only. It is
explicitly told not to answer the medical question, diagnose anything,
or add/remove information — its only job is a faithful translation
that Brain 2 can reason over.

**Brain 2 — Medical reasoning** (`medgemma1.5:4b`)
Receives the English translation and produces an educational,
safety-conscious medical response: possible causes framed as
possibilities (not confirmed diagnoses), warning signs, and a
recommendation to seek professional/emergency care when warranted. It
never mentions Ollama, Gemma, or the pipeline itself.

**Brain 3 — English → Cebuano** (`gemma4:e4b`)
Translates Brain 2's English response into natural Cebuano without
summarizing, softening, or removing any medical warnings, numbers, or
recommendations.

The full text of all three prompts lives in `prompts/` and is loaded
once when the app starts (`brain.py`'s `_load_prompt` calls at import
time) rather than being re-read from disk on every message.

---

## 5. How Ollama communicates with the app

`brain.py` uses the official `ollama` Python library, which talks to
the local Ollama server (normally `http://localhost:11434`) — never
the public internet. Each of the three brains is one call to
`ollama.chat(model=..., messages=[system, user], keep_alive=...)`.

---

## 6. Conversation history

`brain.ConversationHistory` keeps only the most recent
`MAX_HISTORY_TURNS` (default: 3) exchanges. Each stored turn contains
the Cebuano question, its English translation, and the Cebuano reply.
When translating a new message, a short block of prior English
questions is passed alongside it, so Brain 1 can resolve a follow-up
like *"Sakit gihapon hangtod karon"* using the previous symptom as
context — without ever sending the entire conversation history (which
would slow down all three model calls) or growing without bound.

To change how much context is remembered, adjust `MAX_HISTORY_TURNS`
at the top of `brain.py`.

---

## 7. Model keep-alive and performance

Every user message triggers **three** model calls, so keeping models
loaded in memory between requests matters a lot. `brain.py` passes
`keep_alive="10m"` on every Ollama call, which tells Ollama to keep the
model resident for 10 minutes after a response instead of unloading it
immediately — avoiding a multi-second reload penalty on the very next
message. Adjust `KEEP_ALIVE` in `brain.py` if you want a longer/shorter
window.

Other efficiency choices already built in:
- The smaller `gemma4:e4b` model is used for both translation stages.
- Conversation history sent to the models is capped and trimmed.
- Prompt files are loaded once, not per-request.
- The UI runs each pipeline call on a background thread so it stays
  responsive while Ollama generates a response.

---

## 8. Error handling

`brain.py` wraps every Ollama call and raises a `CebuanoDoctorError`
with a friendly Cebuano message for:

- Ollama not running / unreachable (`ConnectionError`)
- A model not installed / not found (`ollama.ResponseError`, 404)
- A model timing out (`TimeoutError`)
- An unexpected or empty model response
- An empty user message
- Any other unexpected exception (final catch-all, so the app never
  crashes or shows a raw traceback)

`main.py` catches `CebuanoDoctorError` and displays only
`friendly_message_cebuano` in the chat window; the technical detail is
available in `e.technical_detail` for logging/debugging but is never
shown to a normal user.

---

## 9. How intermediate results stay hidden

`main.py` only ever calls `brain.process_cebuano_message(...)` and
reads `result["reply"]`. The English translation and raw MedGemma
output exist only inside `brain.py`'s local variables during that one
function call — they are never returned to the UI unless debug mode is
on, and even then they're placed under a separate `result["debug"]`
key that the normal chat bubble never reads.

---

## 10. Debug / evaluation mode

For grading and evaluation, you can inspect every intermediate step.

**To enable it:** open `main.py` and set

```python
DEBUG_MODE = True
```

near the top of the file, then restart the app. When enabled, after
each Cebuano reply the chat window will also print (in a muted, italic
style):

- the English translation of your question,
- MedGemma's raw English response,
- per-stage timings (`cebuano_to_english_sec`, `medical_response_sec`,
  `english_to_cebuano_sec`, `total_sec`).

A **"Debug: View Prompt"** panel also appears above the input box,
with a dropdown listing every `.txt` file currently in `prompts/` and
a **View Prompt** button. Selecting a file and clicking the button
opens a popup showing that prompt's exact raw content — useful for
confirming during evaluation that the application is really using the
prompts you wrote, not a paraphrased or hard-coded version. If the
`prompts/` folder is empty or missing, the button shows the error
**"There are no prompts"** instead of opening a popup. This panel is
built by `main.py`'s `_build_debug_panel()`/`_on_view_prompt()` and
reads live from disk via `brain.list_prompt_files()` and
`brain.get_prompt_content()`, so it always reflects whatever files are
actually in the folder at the moment.

**Debug mode must stay `False` for normal end users** — it is off by
default.

This is intended to support the assignment's evaluation criteria:

- A. Cebuano → English translation accuracy
- B. Medical response quality
- C. English → Cebuano translation accuracy
- D. Response time (from the printed timings)
- E. Medical safety
- F. Completeness
- G. Naturalness/readability of the Cebuano output

Run at least 5 healthcare prompts (ideally provided by others, not
written by you) with debug mode on, and record the intermediate output
for your evaluation write-up.

---

## 11. Safety limitations

Cebuano Doctor is an **AI educational healthcare assistant**, not a
licensed medical professional and not a substitute for one:

- It does not diagnose with certainty; possible causes are presented
  as possibilities.
- It is instructed to flag warning signs and recommend urgent/emergency
  care when the described symptoms reasonably warrant it.
- It does not prescribe individualized medication or dosages.
- All three internal prompts explicitly forbid inventing patient
  information (symptoms, history, test results, medications) that
  wasn't provided.

For any real medical concern, especially anything urgent, consult a
qualified healthcare professional or emergency services.