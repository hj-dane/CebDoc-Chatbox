"""
main.py
-------
This file is the user interface for Cebuano Doctor.

It shows a simple chatbox window (built with Tkinter, which comes
included with Python -- no extra UI framework needed), accepts a
Cebuano healthcare question, and displays the Cebuano answer.

main.py does NOT talk to Ollama directly and does NOT know about
Gemma 4, MedGemma, or the three internal prompts. It only calls
brain.process_cebuano_message(...) and displays whatever Cebuano
text comes back. That keeps the internal pipeline completely hidden
from the normal user.

A background thread is used for each pipeline call so the window
never freezes while the three models are generating a response.
"""

import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk

import brain


# ---------------------------------------------------------------------------
# DEVELOPER SWITCH
# ---------------------------------------------------------------------------
# Set this to True only while developing/evaluating the assignment.
# It must stay False for a normal end user.
DEBUG_MODE = False


class CebuanoDoctorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cebuano Doctor")
        self.root.geometry("520x640")
        self.root.minsize(420, 480)

        brain.set_debug_mode(DEBUG_MODE)
        self.history = brain.ConversationHistory()

        # Track whether a message is currently being processed, so we
        # can disable the Send button and avoid overlapping requests.
        self._processing = False

        self._build_widgets()

    # -----------------------------------------------------------------
    # UI LAYOUT
    # -----------------------------------------------------------------

    def _build_widgets(self):
        header = tk.Label(
            self.root,
            text="CEBUANO DOCTOR",
            font=("Segoe UI", 16, "bold"),
            pady=10,
        )
        header.pack(fill="x")

        subheader = tk.Label(
            self.root,
            text="AI nga edukasyonal nga katabang sa panglawas (dili puli sa doktor)",
            font=("Segoe UI", 9),
            fg="gray30",
        )
        subheader.pack(fill="x", pady=(0, 5))

        # Scrollable chat history area. Read-only from the user's side --
        # we only ever insert text into it from code.
        self.chat_area = scrolledtext.ScrolledText(
            self.root,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 10),
            padx=8,
            pady=8,
        )
        self.chat_area.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        # Text tags so user / doctor / status messages look different.
        self.chat_area.tag_configure("user_label", foreground="#1a5fb4", font=("Segoe UI", 10, "bold"))
        self.chat_area.tag_configure("doctor_label", foreground="#26a269", font=("Segoe UI", 10, "bold"))
        self.chat_area.tag_configure("status", foreground="gray50", font=("Segoe UI", 9, "italic"))
        self.chat_area.tag_configure("error", foreground="#c01c28")

        # Temporary status line (e.g. "Gisabtan ang imong pangutana...").
        self.status_var = tk.StringVar(value="")
        self.status_label = tk.Label(
            self.root, textvariable=self.status_var, fg="gray40", font=("Segoe UI", 9, "italic")
        )
        self.status_label.pack(fill="x", padx=10)

        # Bottom input row: text entry + Send button.
        input_frame = tk.Frame(self.root)
        input_frame.pack(fill="x", padx=10, pady=10)

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(input_frame, textvariable=self.input_var, font=("Segoe UI", 11))
        self.input_entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.input_entry.bind("<Return>", lambda event: self._on_send())
        self.input_entry.insert(0, "")
        self.input_entry.focus_set()

        self.send_button = tk.Button(
            input_frame, text="Ipadala", width=10, command=self._on_send
        )
        self.send_button.pack(side="left", padx=(8, 0))

        # Debug-only panel for browsing the internal prompt files. This is
        # never built (and never visible) unless DEBUG_MODE is True, so a
        # normal user never sees it.
        if DEBUG_MODE:
            self._build_debug_panel()

        self._append_placeholder_prompt()

    def _build_debug_panel(self):
        """
        Debug/evaluation panel: lets the developer pick one of the internal
        prompt files and view its raw content. Only built when DEBUG_MODE
        is True.
        """
        panel = tk.LabelFrame(self.root, text="Debug: View Prompt", padx=8, pady=8)
        panel.pack(fill="x", padx=10, pady=(0, 5))

        row = tk.Frame(panel)
        row.pack(fill="x")

        prompt_files = brain.list_prompt_files()

        self.prompt_choice_var = tk.StringVar()
        self.prompt_dropdown = ttk.Combobox(
            row,
            textvariable=self.prompt_choice_var,
            values=prompt_files,
            state="readonly" if prompt_files else "disabled",
            width=28,
        )
        if prompt_files:
            self.prompt_dropdown.current(0)
        self.prompt_dropdown.pack(side="left", fill="x", expand=True)

        view_button = tk.Button(row, text="View Prompt", command=self._on_view_prompt)
        view_button.pack(side="left", padx=(8, 0))

    def _on_view_prompt(self):
        """Show the selected prompt file's raw content in a popup window."""
        prompt_files = brain.list_prompt_files()

        if not prompt_files:
            messagebox.showerror("Debug", "There are no prompts")
            return

        selected = self.prompt_choice_var.get()
        if not selected:
            messagebox.showerror("Debug", "There are no prompts")
            return

        try:
            content = brain.get_prompt_content(selected)
        except brain.CebuanoDoctorError as e:
            messagebox.showerror("Debug", e.friendly_message_cebuano)
            return

        popup = tk.Toplevel(self.root)
        popup.title(f"Prompt: {selected}")
        popup.geometry("480x420")

        text_area = scrolledtext.ScrolledText(popup, wrap="word", font=("Consolas", 10))
        text_area.pack(fill="both", expand=True, padx=8, pady=8)
        text_area.insert("1.0", content)
        text_area.configure(state="disabled")

    def _append_placeholder_prompt(self):
        self._append_chat(
            "Isulat ang imong pangutana sa Cebuano bahin sa imong panglawas.\n",
            tag="status",
        )

    # -----------------------------------------------------------------
    # CHAT DISPLAY HELPERS
    # -----------------------------------------------------------------

    def _append_chat(self, text, tag=None):
        self.chat_area.configure(state="normal")
        if tag:
            self.chat_area.insert("end", text, tag)
        else:
            self.chat_area.insert("end", text)
        self.chat_area.configure(state="disabled")
        self.chat_area.see("end")

    def _append_user_message(self, text):
        self._append_chat("Ikaw\n", tag="user_label")
        self._append_chat(f"{text}\n\n")

    def _append_doctor_message(self, text):
        self._append_chat("Doctor\n", tag="doctor_label")
        self._append_chat(f"{text}\n\n")

    def _append_error_message(self, text):
        self._append_chat(f"{text}\n\n", tag="error")

    # -----------------------------------------------------------------
    # SEND HANDLING
    # -----------------------------------------------------------------

    def _on_send(self):
        if self._processing:
            return  # ignore extra clicks while a request is in flight

        user_text = self.input_var.get().strip()
        if not user_text:
            return  # empty message -- nothing to send

        self.input_var.set("")
        self._append_user_message(user_text)

        self._set_processing(True)

        # Run the (slow) pipeline on a background thread so the Tkinter
        # UI loop stays responsive and never freezes.
        thread = threading.Thread(
            target=self._run_pipeline, args=(user_text,), daemon=True
        )
        thread.start()

    def _set_processing(self, is_processing):
        self._processing = is_processing
        state = "disabled" if is_processing else "normal"
        self.send_button.configure(state=state)
        self.input_entry.configure(state=state)
        if not is_processing:
            self.status_var.set("")
            self.input_entry.focus_set()

    def _set_status(self, message):
        # Called from the background thread; Tkinter widget updates must
        # happen on the main thread, so we hop back over with `after`.
        self.root.after(0, lambda: self.status_var.set(message))

    def _run_pipeline(self, user_text):
        try:
            result = brain.process_cebuano_message(
                user_text, self.history, on_status=self._set_status
            )
        except brain.CebuanoDoctorError as e:
            self.root.after(0, lambda: self._handle_error(e.friendly_message_cebuano))
            return
        except Exception as e:
            # Final safety net -- never let an unexpected exception
            # crash the app or show a raw traceback to the user.
            self.root.after(
                0,
                lambda: self._handle_error(
                    "Pasensya, adunay wala damha nga problema. Palihog sulayi pag-usab."
                ),
            )
            return

        self.root.after(0, lambda: self._handle_success(result))

    def _handle_success(self, result):
        self._append_doctor_message(result["reply"])
        if brain.is_debug_mode() and "debug" in result:
            debug = result["debug"]
            debug_text = (
                "[DEBUG] English translation: {english_translation}\n"
                "[DEBUG] English medical response: {english_medical_response}\n"
                "[DEBUG] Timings: {timings}\n"
            ).format(**debug)
            self._append_chat(debug_text, tag="status")
        self._set_processing(False)

    def _handle_error(self, friendly_message):
        self._append_error_message(friendly_message)
        self._set_processing(False)


def main():
    root = tk.Tk()
    app = CebuanoDoctorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()