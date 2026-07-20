import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from gui_common import FileListPanel, LogPanel, ScrollableFrame, make_browse_row, resolve_training_python


def validate_job(job):
    errors = []
    images = job.get("images", [])
    layers = job.get("layers", [])
    outdir = job.get("outdir", "")

    if not images:
        errors.append("Add at least one image.")
    if not layers or (images and len(layers) % len(images) != 0):
        errors.append(f"Layer masks ({len(layers)}) must be a non-zero multiple of image count ({len(images)}).")
    if not outdir:
        errors.append("Choose an output directory.")

    return errors


def build_derive_cmd(job, script_path, python_executable):
    return [
        python_executable, "-u", script_path,
        "--images", *job["images"],
        "--layer-masks", *job["layers"],
        "--output-dir", job["outdir"],
    ]


class DeriveBgMaskGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Derive Background Mask")
        self.resizable(True, True)
        self.geometry("640x640")

        scroll = ScrollableFrame(self)
        scroll.pack(fill='both', expand=True)
        body = scroll.body

        tk.Label(
            body, wraplength=600, justify='left', padx=10, pady=6,
            text=("Writes {image_stem}_bg.png = pixel-wise inverse of the union of the given "
                  "layer mask(s) per image. Correct when background means \"everything not "
                  "covered by these layer(s)\" (e.g. a staff-only classifier) — not appropriate "
                  "if some pixels should be excluded from every mask.")
        ).pack(fill='x')

        self._images_panel = FileListPanel(body, "Input Images", add_label="+ Add Image")
        self._images_panel.pack(fill='x', padx=10, pady=2)

        self._layer_panel = FileListPanel(body, "Layer Masks (flat list, N-images x L-layers, RGBA PNG)",
                                          add_label="+ Add Layer Mask")
        self._layer_panel.pack(fill='x', padx=10, pady=2)

        outdir_frame = tk.LabelFrame(body, text="Output Directory", padx=8, pady=8)
        outdir_frame.pack(fill='x', padx=10, pady=4)
        self._outdir_var = tk.StringVar()
        make_browse_row(outdir_frame, "Output dir:", self._outdir_var, filedialog.askdirectory, readonly=True)

        self._run_btn = tk.Button(body, text="Derive Background Masks", font=('TkDefaultFont', 11, 'bold'),
                                  command=self._on_run)
        self._run_btn.pack(pady=8)

        self._log = LogPanel(body)
        self._log.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        self._log.poll(self)

        scroll.enable_mousewheel()

    def _collect_job(self):
        return {
            "images": self._images_panel.get_paths(),
            "layers": self._layer_panel.get_paths(),
            "outdir": self._outdir_var.get().strip(),
        }

    def _on_run(self):
        job = self._collect_job()
        errors = validate_job(job)
        if errors:
            messagebox.showerror("Validation error", "\n".join(errors))
            return

        self._run_btn.config(state='disabled')
        self._log.write("\n--- Deriving background masks ---\n")
        threading.Thread(target=self._run_derive, args=(job,), daemon=True).start()

    def _run_derive(self, job):
        script = str(Path(__file__).parent.parent / "derive_background_mask.py")
        cmd = build_derive_cmd(job, script, resolve_training_python())
        self._log.write(f"Using Python: {cmd[0]}\n")

        status = "Done"
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(proc.stdout.readline, ''):
                self._log.write(line)
            proc.stdout.close()
            proc.wait()
            if proc.returncode != 0:
                status = f"Failed (exit code {proc.returncode})"
        except Exception as exc:
            status = f"Error: {exc}"

        self._log.write(f"\n--- {status} ---\n")
        self.after(0, self._run_btn.config, {'state': 'normal'})


if __name__ == '__main__':
    app = DeriveBgMaskGUI()
    app.mainloop()
