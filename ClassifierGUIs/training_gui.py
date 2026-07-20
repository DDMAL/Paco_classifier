import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from gui_common import (FileListPanel, LogPanel, MODEL_FILETYPES, ScrollableFrame, build_param_grid,
                         build_train_cmd, make_browse_row, resolve_training_python, validate_job)


class TrainingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calvo Independent Training")
        self.resizable(True, True)
        self.geometry("720x760")

        scroll = ScrollableFrame(self)
        scroll.pack(fill='both', expand=True)
        body = scroll.body

        self._notebook = ttk.Notebook(body)
        self._notebook.pack(fill='x', padx=10, pady=(10, 4))

        self._build_inputs_tab()
        self._build_params_tab()

        self._run_btn = tk.Button(body, text="Run Training", font=('TkDefaultFont', 11, 'bold'),
                                  command=self._on_run)
        self._run_btn.pack(pady=8)

        self._log = LogPanel(body)
        self._log.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        self._log.poll(self)

        scroll.enable_mousewheel()

    def _build_inputs_tab(self):
        tab = tk.Frame(self._notebook)
        self._notebook.add(tab, text="Inputs")

        self._images_panel = FileListPanel(tab, "Input Images", add_label="+ Add Image")
        self._images_panel.pack(fill='x', pady=2)

        self._bg_panel = FileListPanel(tab, "Background Masks (optional, one per image, RGBA PNG)", add_label="+ Add BG Mask")
        self._bg_panel.pack(fill='x', pady=2)

        self._layer_panel = FileListPanel(tab, "Layer Masks (flat list, N-images x L-layers, RGBA PNG)",
                                          add_label="+ Add Layer Mask")
        self._layer_panel.pack(fill='x', pady=2)

        self._region_panel = FileListPanel(tab, "Region Masks (optional, one per image, RGBA PNG)",
                                           add_label="+ Add Region Mask")
        self._region_panel.pack(fill='x', pady=2)

        self._pretrained_panel = FileListPanel(
            tab, "Pretrained Models (optional, one per label: background then layers, .h5)",
            add_label="+ Add Pretrained Model", filetypes=MODEL_FILETYPES)
        self._pretrained_panel.pack(fill='x', pady=2)

    def _build_params_tab(self):
        tab = tk.Frame(self._notebook)
        self._notebook.add(tab, text="Parameters & Output")

        self._param_vars = build_param_grid(tab)

        outdir_frame = tk.LabelFrame(tab, text="Output Directory", padx=8, pady=8)
        outdir_frame.pack(fill='x', padx=10, pady=4)
        self._outdir_var = tk.StringVar()
        make_browse_row(outdir_frame, "Output dir:", self._outdir_var, filedialog.askdirectory, readonly=True)

    def _collect_job(self):
        return {
            "images": self._images_panel.get_paths(),
            "bg": self._bg_panel.get_paths(),
            "layers": self._layer_panel.get_paths(),
            "regions": self._region_panel.get_paths(),
            "pretrained": self._pretrained_panel.get_paths(),
            "outdir": self._outdir_var.get().strip(),
            "params": {key: var.get() for key, var in self._param_vars.items()},
        }

    def _on_run(self):
        job = self._collect_job()
        errors = validate_job(job)
        if errors:
            messagebox.showerror("Validation error", "\n".join(errors))
            return

        self._run_btn.config(state='disabled')
        self._log.write("\n--- Starting training ---\n")
        threading.Thread(target=self._run_training, args=(job,), daemon=True).start()

    def _run_training(self, job):
        script = str(Path(__file__).parent.parent / "calvo_independent_train.py")
        cmd = build_train_cmd(job, script, resolve_training_python())
        self._log.write(f"Using training Python: {cmd[0]}\n")

        status = "Training complete"
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(proc.stdout.readline, ''):
                self._log.write(line)
            proc.stdout.close()
            proc.wait()
            if proc.returncode != 0:
                status = f"Training failed (exit code {proc.returncode})"
        except Exception as exc:
            status = f"Error: {exc}"

        self._log.write(f"\n--- {status} ---\n")
        self.after(0, self._run_btn.config, {'state': 'normal'})


if __name__ == '__main__':
    app = TrainingGUI()
    app.mainloop()
