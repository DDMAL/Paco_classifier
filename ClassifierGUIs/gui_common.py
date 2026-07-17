"""Shared tkinter widgets and helpers for the Paco Classifier training GUIs."""
import queue
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk

IMAGE_FILETYPES = [("Image / PNG", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                    ("All files", "*.*")]
MODEL_FILETYPES = [("Model files", "*.h5"), ("All files", "*.*")]


def resolve_training_python():
    """Path to the Python that has the project's ML deps installed (TensorFlow, etc.).

    The GUI process may be launched under a different interpreter than the one used to
    actually run calvo_independent_train.py (e.g. one with a working Tk on macOS, since
    the project's own .venv bundles a broken system Tcl/Tk) — this looks for the repo's
    own .venv first and falls back to sys.executable.
    """
    repo_venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python3"
    if repo_venv_python.exists():
        return str(repo_venv_python)
    return sys.executable


class FileListPanel(tk.LabelFrame):
    """A labeled panel wrapping a Listbox of file paths plus Add/Remove buttons.

    Replaces the old per-row Frame+Entry+Browse+x pattern with a single, compact Listbox.
    """

    def __init__(self, parent, title, add_label="+ Add", multiple=True,
                 filetypes=IMAGE_FILETYPES, height=4):
        super().__init__(parent, text=title, padx=8, pady=8)
        self._multiple = multiple
        self._filetypes = filetypes

        list_frame = tk.Frame(self)
        list_frame.pack(fill='x')

        self.listbox = tk.Listbox(list_frame, height=height, selectmode='extended')
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        btn_row = tk.Frame(self)
        btn_row.pack(fill='x', pady=(4, 0))
        tk.Button(btn_row, text=add_label, command=self._add).pack(side='left')
        tk.Button(btn_row, text="Remove Selected", command=self._remove_selected).pack(side='left', padx=(4, 0))

    def _add(self):
        if self._multiple:
            paths = filedialog.askopenfilenames(filetypes=self._filetypes)
        else:
            path = filedialog.askopenfilename(filetypes=self._filetypes)
            paths = [path] if path else []
        for path in paths:
            if path:
                self.listbox.insert(tk.END, path)

    def _remove_selected(self):
        for idx in reversed(self.listbox.curselection()):
            self.listbox.delete(idx)

    def get_paths(self):
        return list(self.listbox.get(0, tk.END))

    def set_paths(self, paths):
        self.listbox.delete(0, tk.END)
        for path in paths:
            self.listbox.insert(tk.END, path)

    def clear(self):
        self.listbox.delete(0, tk.END)


def make_browse_row(parent, label, var, dialog_cmd, width=50):
    row = tk.Frame(parent)
    row.pack(fill='x', pady=2)
    tk.Label(row, text=label, width=20, anchor='w').pack(side='left')
    tk.Entry(row, textvariable=var, width=width).pack(side='left', fill='x', expand=True)
    tk.Button(row, text="Browse", command=lambda: var.set(dialog_cmd() or var.get())).pack(side='left')
    return row


PARAM_SPECS = [
    ("height", "Height", 256, 1, 9999, 1.0),
    ("width", "Width", 256, 1, 9999, 1.0),
    ("epochs", "Epochs", 50, 1, 9999, 1.0),
    ("max_samples_per_class", "Max Samples/Class", 1000, 1, 999999, 1.0),
    ("batch_size", "Batch Size", 16, 1, 4096, 1.0),
    ("ram_limit", "RAM Limit (GB)", 4.0, 0.1, 999.0, 0.5),
]


def build_param_grid(parent):
    """Builds the Height/Width/Epochs/Max-Samples/Batch-Size/RAM-Limit grid.

    Returns a dict keyed by the PARAM_SPECS key (e.g. "height") -> tk.StringVar.
    """
    frame = tk.LabelFrame(parent, text="Parameters", padx=8, pady=8)
    frame.pack(fill='x', padx=10, pady=4)
    grid = tk.Frame(frame)
    grid.pack(anchor='w')

    variables = {}
    for row, (key, label, default, lo, hi, inc) in enumerate(PARAM_SPECS):
        var = tk.StringVar(value=str(default))
        tk.Label(grid, text=label + ":", anchor='w', width=22).grid(row=row, column=0, sticky='w', pady=2)
        ttk.Spinbox(grid, textvariable=var, from_=lo, to=hi, increment=inc, width=10).grid(row=row, column=1, sticky='w')
        variables[key] = var
    return variables


def validate_job(job):
    """job: dict with images/bg/layers/regions/pretrained (lists of str) + outdir (str).

    Returns a list of error strings (empty if valid).
    """
    errors = []
    images = job.get("images", [])
    bg = job.get("bg", [])
    layers = job.get("layers", [])
    regions = job.get("regions", [])
    pretrained = job.get("pretrained", [])
    outdir = job.get("outdir", "")

    if not images:
        errors.append("Add at least one image.")
    if len(bg) != len(images):
        errors.append(f"Background masks ({len(bg)}) must equal image count ({len(images)}).")
    if not layers or (images and len(layers) % len(images) != 0):
        errors.append(f"Layer masks ({len(layers)}) must be a non-zero multiple of image count ({len(images)}).")
    if regions and len(regions) != len(images):
        errors.append(f"Region masks ({len(regions)}) must equal image count ({len(images)}) or be empty.")
    if not outdir:
        errors.append("Choose an output directory.")

    num_labels = 1 + (len(layers) // len(images) if images and layers else 0)
    if pretrained and len(pretrained) != num_labels:
        errors.append(
            f"Pretrained models ({len(pretrained)}) must equal number of labels "
            f"({num_labels}: background + layers) or be empty.")

    return errors


def build_train_cmd(job, script_path, python_executable):
    """Builds the calvo_independent_train.py subprocess argv for one job dict."""
    cmd = [
        python_executable, "-u", script_path,
        "--images", *job["images"],
        "--background-mask", *job["bg"],
        "--layer-masks", *job["layers"],
        "--output-dir", job["outdir"],
        "--height", str(job["params"]["height"]),
        "--width", str(job["params"]["width"]),
        "--epochs", str(job["params"]["epochs"]),
        "--max-samples-per-class", str(job["params"]["max_samples_per_class"]),
        "--batch-size", str(job["params"]["batch_size"]),
        "--ram-limit", str(job["params"]["ram_limit"]),
    ]
    if job.get("regions"):
        cmd += ["--regions-mask", *job["regions"]]
    if job.get("pretrained"):
        cmd += ["--pretrained-models", *job["pretrained"]]
    return cmd


class LogPanel(tk.LabelFrame):
    """A scrolled, read-only log box fed from a background thread via a queue.Queue."""

    def __init__(self, parent, height=14):
        super().__init__(parent, text="Log", padx=8, pady=8)
        self._queue = queue.Queue()
        self.text = scrolledtext.ScrolledText(self, height=height, state='disabled',
                                              wrap='word', font=('Courier', 10))
        self.text.pack(fill='both', expand=True)

    def write(self, text):
        """Thread-safe: call from any thread, queues the text for the poll loop to display."""
        self._queue.put(text)

    def _write_now(self, text):
        self.text.config(state='normal')
        self.text.insert(tk.END, text)
        self.text.see(tk.END)
        self.text.config(state='disabled')

    def poll(self, root, interval_ms=100):
        while True:
            try:
                msg = self._queue.get_nowait()
            except queue.Empty:
                break
            self._write_now(msg)
        root.after(interval_ms, lambda: self.poll(root, interval_ms))
