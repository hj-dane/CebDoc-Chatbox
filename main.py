import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import scrolledtext, messagebox, ttk

import brain


# ---------------------------------------------------------------------------
# DEVELOPER SWITCH
# ---------------------------------------------------------------------------
DEBUG_MODE = False 


class CebuanoDoctorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cebuano Doctor")
        self.root.geometry("640x780")
        self.root.minsize(500, 600)
        self.root.configure(bg="#f8f9fa")

        # Cache font helper to calculate bubble widths quickly
        self.bubble_font = tkfont.Font(family="Segoe UI", size=15)

        brain.set_debug_mode(DEBUG_MODE)
        self.history = brain.ConversationHistory()

        self._processing = False

        self._build_widgets()

        # Pre-warm Ollama models in background thread on startup
        threading.Thread(target=brain.warmup_models, daemon=True).start()

    # -----------------------------------------------------------------
    # UI LAYOUT
    # -----------------------------------------------------------------

    def _build_widgets(self):
        # 1. FIXED TOP HEADER
        header_frame = tk.Frame(self.root, bg="#f8f9fa", pady=10)
        header_frame.pack(side="top", fill="x")

        header = tk.Label(
            header_frame,
            text="CEBUANO DOCTOR",
            font=("Segoe UI", 21, "bold"),
            fg="#0d6efd",
            bg="#f8f9fa",
        )
        header.pack(fill="x")

        subheader = tk.Label(
            header_frame,
            text="AI nga edukasyonal nga katabang sa panglawas (dili puli sa doktor)",
            font=("Segoe UI", 11),
            fg="#6c757d",
            bg="#f8f9fa",
        )
        subheader.pack(fill="x", pady=(2, 0))

        # 2. FIXED BOTTOM INPUT DOCK
        input_frame = tk.Frame(self.root, bg="#f8f9fa")
        input_frame.pack(side="bottom", fill="x", padx=15, pady=15)

        pill_container = tk.Frame(input_frame, bg="#f8f9fa")
        pill_container.pack(side="left", fill="x", expand=True, padx=(0, 10))

        pill_canvas = tk.Canvas(
            pill_container,
            height=50,
            bg="#f8f9fa",
            highlightthickness=0,
            bd=0
        )
        pill_canvas.pack(fill="x", expand=True)

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(
            pill_canvas,
            textvariable=self.input_var,
            font=("Segoe UI", 15),
            bg="#ffffff",
            fg="#000000",
            relief="flat",
            bd=0,
            highlightthickness=0
        )

        def _draw_pill(event):
            w = event.width
            h = event.height
            r = h // 2
            pill_canvas.delete("all")
            pill_canvas.create_arc(0, 0, h, h, start=90, extent=180, fill="#ffffff", outline="#ced4da")
            pill_canvas.create_arc(w - h, 0, w, h, start=-90, extent=180, fill="#ffffff", outline="#ced4da")
            pill_canvas.create_rectangle(r, 0, w - r, h, fill="#ffffff", outline="#ffffff")
            pill_canvas.create_line(r, 0, w - r, 0, fill="#ced4da")
            pill_canvas.create_line(r, h - 1, w - r, h - 1, fill="#ced4da")
            pill_canvas.create_window(r, h // 2, window=self.input_entry, width=w - (r * 2), anchor="w")

        pill_canvas.bind("<Configure>", _draw_pill)

        self.input_entry.bind("<Return>", lambda event: self._on_send())
        self.input_entry.focus_set()

        self.send_button = tk.Button(
            input_frame, 
            text="Ipadala", 
            width=10, 
            font=("Segoe UI", 13, "bold"),
            bg="#0d6efd",
            fg="white",
            activebackground="#0b5ed7",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            command=self._on_send
        )
        self.send_button.pack(side="left", ipady=6)

        self.status_var = tk.StringVar(value="")
        self.status_label = tk.Label(
            self.root, 
            textvariable=self.status_var, 
            fg="#6c757d", 
            bg="#f8f9fa",
            font=("Segoe UI", 12, "italic"),
            wraplength=460,
            justify="left",
        )
        self.status_label.pack(side="bottom", fill="x", padx=15, pady=(0, 5))

        if DEBUG_MODE:
            self._build_debug_panel()

        # 3. MIDDLE CHAT AREA
        self.chat_area = scrolledtext.ScrolledText(
            self.root,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 15),
            padx=12,
            pady=12,
            bg="#ffffff",
            relief="flat",
            bd=1,
            height=12
        )
        self.chat_area.pack(side="top", fill="both", expand=True, padx=15, pady=(0, 5))

        self.chat_area.tag_configure("user_align", justify="right")
        self.chat_area.tag_configure("doctor_align", justify="left")
        self.chat_area.tag_configure("user_label", foreground="#0b5ed7", font=("Segoe UI", 12, "bold"))
        self.chat_area.tag_configure("doctor_label", foreground="#0a58ca", font=("Segoe UI", 12, "bold"))
        self.chat_area.tag_configure("status", foreground="#6c757d", font=("Segoe UI", 13, "italic"))
        self.chat_area.tag_configure("error", foreground="#dc3545", font=("Segoe UI", 14, "bold"))

        self._append_placeholder_prompt()

    def _build_debug_panel(self):
        panel = tk.LabelFrame(self.root, text="Debug: View Prompt", padx=8, pady=8, bg="#f8f9fa")
        panel.pack(side="bottom", fill="x", padx=15, pady=(0, 5))

        row = tk.Frame(panel, bg="#f8f9fa")
        row.pack(fill="x")

        prompt_files = brain.list_prompt_files()

        self.prompt_choice_var = tk.StringVar()
        self.prompt_dropdown = ttk.Combobox(
            row,
            textvariable=self.prompt_choice_var,
            values=prompt_files,
            state="readonly" if prompt_files else "disabled",
            width=28,
            font=("Segoe UI", 12)
        )
        if prompt_files:
            self.prompt_dropdown.current(0)
        self.prompt_dropdown.pack(side="left", fill="x", expand=True)

        view_button = tk.Button(
            row, 
            text="View Prompt", 
            font=("Segoe UI", 11),
            command=self._on_view_prompt
        )
        view_button.pack(side="left", padx=(8, 0))

    def _on_view_prompt(self):
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
        popup.geometry("550x500")

        text_area = scrolledtext.ScrolledText(popup, wrap="word", font=("Consolas", 12))
        text_area.pack(fill="both", expand=True, padx=10, pady=10)
        text_area.insert("1.0", content)
        text_area.configure(state="disabled")

    def _append_placeholder_prompt(self):
        self._append_chat(
            "Isulat ang imong pangutana sa Cebuano bahin sa imong panglawas.\n\n",
            tag="status",
        )

    # -----------------------------------------------------------------
    # FAST CHAT BUBBLE RENDERING
    # -----------------------------------------------------------------

    def _create_rounded_bubble(self, parent_widget, text, bg_color, fg_color, max_width=400, radius=14):
        """Fast creation of rounded bubbles using cached tkfont measurement."""
        lines = text.split("\n")
        calc_lines = []
        for line in lines:
            if not line:
                calc_lines.append("")
                continue
            words = line.split(" ")
            curr_line = ""
            for w in words:
                test_line = f"{curr_line} {w}".strip()
                if self.bubble_font.measure(test_line) <= max_width:
                    curr_line = test_line
                else:
                    calc_lines.append(curr_line)
                    curr_line = w
            if curr_line:
                calc_lines.append(curr_line)

        wrapped_text = "\n".join(calc_lines)
        text_w = max(self.bubble_font.measure(l) for l in calc_lines) if calc_lines else 10
        text_h = len(calc_lines) * (self.bubble_font.metrics("linespace") + 2)

        pad_x, pad_y = 16, 12
        w = text_w + (pad_x * 2)
        h = text_h + (pad_y * 2)

        canvas = tk.Canvas(
            parent_widget,
            width=w,
            height=h,
            bg="#ffffff",
            highlightthickness=0,
            bd=0
        )

        r = radius * 2
        canvas.create_arc(0, 0, r, r, start=90, extent=90, fill=bg_color, outline=bg_color)
        canvas.create_arc(w - r, 0, w, r, start=0, extent=90, fill=bg_color, outline=bg_color)
        canvas.create_arc(w - r, h - r, w, h, start=270, extent=90, fill=bg_color, outline=bg_color)
        canvas.create_arc(0, h - r, r, h, start=180, extent=90, fill=bg_color, outline=bg_color)

        canvas.create_rectangle(radius, 0, w - radius, h, fill=bg_color, outline=bg_color)
        canvas.create_rectangle(0, radius, w, h - radius, fill=bg_color, outline=bg_color)

        canvas.create_text(
            pad_x, pad_y,
            text=wrapped_text,
            font=self.bubble_font,
            fill=fg_color,
            anchor="nw"
        )

        return canvas

    def _append_chat(self, text, tag=None):
        self.chat_area.configure(state="normal")
        if tag:
            self.chat_area.insert("end", text, tag)
        else:
            self.chat_area.insert("end", text)
        self.chat_area.configure(state="disabled")
        self.chat_area.see("end")

    def _append_user_message(self, text):
        self.chat_area.configure(state="normal")
        self.chat_area.insert("end", "Ikaw\n", ("user_align", "user_label"))

        chat_width = self.chat_area.winfo_width()
        if chat_width < 100:
            chat_width = 580

        row_frame = tk.Frame(self.chat_area, bg="#ffffff", width=chat_width - 30)

        bubble = self._create_rounded_bubble(
            parent_widget=row_frame,
            text=text, 
            bg_color="#e7f1ff", 
            fg_color="#0c4a6e", 
            radius=14
        )
        row_frame.configure(height=bubble.winfo_reqheight())
        row_frame.pack_propagate(False)
        bubble.pack(side="right", anchor="e")

        self.chat_area.window_create("end", window=row_frame)
        self.chat_area.insert("end", "\n\n", "user_align")
        self.chat_area.configure(state="disabled")
        self.chat_area.see("end")

    def _append_doctor_message(self, text):
        self.chat_area.configure(state="normal")
        self.chat_area.insert("end", "Doctor\n", ("doctor_align", "doctor_label"))
        
        row_frame = tk.Frame(self.chat_area, bg="#ffffff")
        bubble = self._create_rounded_bubble(
            parent_widget=row_frame,
            text=text, 
            bg_color="#0d6efd", 
            fg_color="#ffffff", 
            radius=14
        )
        bubble.pack(side="left")
        
        self.chat_area.window_create("end", window=row_frame)
        self.chat_area.insert("end", "\n\n", "doctor_align")
        self.chat_area.configure(state="disabled")
        self.chat_area.see("end")

    def _append_error_message(self, text):
        self._append_chat(f"{text}\n\n", tag="error")

    # -----------------------------------------------------------------
    # SEND HANDLING
    # -----------------------------------------------------------------

    def _on_send(self):
        if self._processing:
            return

        user_text = self.input_var.get().strip()
        if not user_text:
            return

        self.input_var.set("")
        self._append_user_message(user_text)

        self._set_processing(True)

        thread = threading.Thread(
            target=self._run_pipeline, args=(user_text,), daemon=True
        )
        thread.start()

    def _set_processing(self, is_processing):
        self._processing = is_processing
        state = "disabled" if is_processing else "normal"
        btn_bg = "#6c757d" if is_processing else "#0d6efd"
        self.send_button.configure(state=state, bg=btn_bg)
        self.input_entry.configure(state=state)
        if not is_processing:
            self.status_var.set("")
            self.input_entry.focus_set()

    def _set_status(self, message):
        self.root.after(0, lambda: self.status_var.set(message))

    def _make_stream_chunk_handler(self):
        state = {"text": "", "update_pending": False}

        def _on_chunk(piece):
            state["text"] += piece
            if state["update_pending"]:
                return
            state["update_pending"] = True

            def _flush():
                state["update_pending"] = False
                preview = state["text"]
                if len(preview) > 220:
                    preview = "..." + preview[-220:]
                self.status_var.set(preview)

            self.root.after(0, _flush)

        return _on_chunk

    def _run_pipeline(self, user_text):
        try:
            result = brain.process_cebuano_message(
                user_text,
                self.history,
                on_status=self._set_status,
                on_cebuano_chunk=self._make_stream_chunk_handler(),
            )
        except brain.CebuanoDoctorError as e:
            # Capture the message into a plain string BEFORE scheduling the
            # lambda. self.root.after() runs this lambda later, on the next
            # Tkinter event loop pass -- by then Python has already deleted
            # `e` (exception variables from `except ... as e:` only live
            # inside that block and are auto-removed once it ends), so a
            # lambda that references `e` directly raises NameError instead
            # of showing the actual error message.
            friendly_message = e.friendly_message_cebuano
            self.root.after(0, lambda: self._handle_error(friendly_message))
            return
        except Exception:
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
                "[DEBUG] Timings: {timings}\n\n"
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