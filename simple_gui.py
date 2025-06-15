import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
import numpy as np
from backend.transcription import transcribe_audio


class WhisperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MLX Whisper Transcription")

        # Recording controls
        self.record_btn = ttk.Button(
            root, text="Start Recording", command=self.toggle_recording
        )
        self.record_btn.pack(pady=10)

        # Transcription display
        self.text = tk.Text(root, height=15, width=60)
        self.text.pack(pady=10)

        # Model selection
        self.model_var = tk.StringVar(value="base")
        ttk.Label(root, text="Model:").pack()
        ttk.Combobox(
            root,
            textvariable=self.model_var,
            values=["tiny", "base", "small", "medium", "large"],
        ).pack()

        # Status bar
        self.status = ttk.Label(root, text="Ready")
        self.status.pack(fill=tk.X)

        self.recording = False
        self.frames = []

    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.record_btn.config(text="Stop Recording")
            self.status.config(text="Recording...")
            self.frames = []
            self.stream = sd.InputStream(callback=self.audio_callback)
            self.stream.start()
        else:
            self.recording = False
            self.record_btn.config(text="Start Recording")
            self.stream.stop()
            self.transcribe()

    def audio_callback(self, indata, frames, time, status):
        self.frames.append(indata.copy())

    def transcribe(self):
        try:
            audio = np.concatenate(self.frames)
            result = transcribe_audio(audio, model=self.model_var.get())
            self.text.insert(tk.END, result["text"] + "\n")
            self.status.config(text="Done")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.config(text="Error")


if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperGUI(root)
    root.mainloop()
