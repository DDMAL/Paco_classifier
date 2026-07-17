import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from gui_common import (FileListPanel, LogPanel, MODEL_FILETYPES, build_param_grid,
                         build_train_cmd, make_browse_row, resolve_training_python, validate_job)


class BatchTrainingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calvo Independent Training — Batch Queue")
        self.resizable(True, True)
        self.geometry("820x820")

        self._jobs = []  # list of job dicts, in run order
        self._stop_requested = False

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill='x', padx=10, pady=(10, 4))

        self._build_job_editor_tab()
        self._build_queue_tab()
        self._build_run_controls()

        self._log = LogPanel(self)
        self._log.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        self._log.poll(self)

    # --- job editor tab (mirrors training_gui.py's form) ---

    def _build_job_editor_tab(self):
        tab = tk.Frame(self._notebook)
        self._notebook.add(tab, text="Job Editor")

        self._images_panel = FileListPanel(tab, "Input Images", add_label="+ Add Image", height=3)
        self._images_panel.pack(fill='x', pady=2)

        self._bg_panel = FileListPanel(tab, "Background Masks (one per image)", add_label="+ Add BG Mask", height=3)
        self._bg_panel.pack(fill='x', pady=2)

        self._layer_panel = FileListPanel(tab, "Layer Masks (flat list)", add_label="+ Add Layer Mask", height=3)
        self._layer_panel.pack(fill='x', pady=2)

        self._region_panel = FileListPanel(tab, "Region Masks (optional)", add_label="+ Add Region Mask", height=3)
        self._region_panel.pack(fill='x', pady=2)

        self._pretrained_panel = FileListPanel(tab, "Pretrained Models (optional)", add_label="+ Add Pretrained Model",
                                               filetypes=MODEL_FILETYPES, height=3)
        self._pretrained_panel.pack(fill='x', pady=2)

        self._param_vars = build_param_grid(tab)

        outdir_frame = tk.Frame(tab)
        outdir_frame.pack(fill='x', pady=4)
        self._outdir_var = tk.StringVar()
        make_browse_row(outdir_frame, "Output dir:", self._outdir_var, filedialog.askdirectory)

        btn_row = tk.Frame(tab)
        btn_row.pack(fill='x', pady=(4, 8))
        tk.Button(btn_row, text="Add Job to Queue", command=self._add_job).pack(side='left')
        tk.Button(btn_row, text="Update Selected Job", command=self._update_job).pack(side='left', padx=(4, 0))
        tk.Button(btn_row, text="Clear Form", command=self._clear_form).pack(side='left', padx=(4, 0))

    def _collect_job(self):
        return {
            "images": self._images_panel.get_paths(),
            "bg": self._bg_panel.get_paths(),
            "layers": self._layer_panel.get_paths(),
            "regions": self._region_panel.get_paths(),
            "pretrained": self._pretrained_panel.get_paths(),
            "outdir": self._outdir_var.get().strip(),
            "params": {key: var.get() for key, var in self._param_vars.items()},
            "status": "Pending",
        }

    def _load_job_into_form(self, job):
        self._images_panel.set_paths(job["images"])
        self._bg_panel.set_paths(job["bg"])
        self._layer_panel.set_paths(job["layers"])
        self._region_panel.set_paths(job["regions"])
        self._pretrained_panel.set_paths(job["pretrained"])
        self._outdir_var.set(job["outdir"])
        for key, var in self._param_vars.items():
            var.set(str(job["params"][key]))

    def _clear_form(self):
        for panel in (self._images_panel, self._bg_panel, self._layer_panel,
                      self._region_panel, self._pretrained_panel):
            panel.clear()
        self._outdir_var.set("")

    # --- queue tab ---

    def _build_queue_tab(self):
        tab = tk.Frame(self._notebook)
        self._notebook.add(tab, text="Job Queue")

        tree_frame = tk.Frame(tab)
        tree_frame.pack(fill='both', expand=True, pady=(4, 0))

        columns = ("name", "status")
        self._tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=10)
        self._tree.heading("name", text="Job (Output Dir)")
        self._tree.heading("status", text="Status")
        self._tree.column("name", width=520)
        self._tree.column("status", width=120)
        self._tree.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

        btn_row = tk.Frame(tab)
        btn_row.pack(fill='x', pady=(4, 8))
        tk.Button(btn_row, text="Load Into Editor", command=self._load_selected_into_editor).pack(side='left')
        tk.Button(btn_row, text="Remove Selected", command=self._remove_selected_job).pack(side='left', padx=(4, 0))
        tk.Button(btn_row, text="Move Up", command=lambda: self._move_job(-1)).pack(side='left', padx=(4, 0))
        tk.Button(btn_row, text="Move Down", command=lambda: self._move_job(1)).pack(side='left', padx=(4, 0))

    def _load_selected_into_editor(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        self._load_job_into_form(self._jobs[idx])
        self._notebook.select(0)

    def _add_job(self):
        job = self._collect_job()
        errors = validate_job(job)
        if errors:
            messagebox.showerror("Validation error", "\n".join(errors))
            return
        self._jobs.append(job)
        self._refresh_tree()
        self._clear_form()

    def _update_job(self):
        sel = self._tree.selection()
        if not sel:
            messagebox.showerror("No selection", "Select a job in the Job Queue tab to update.")
            return
        job = self._collect_job()
        errors = validate_job(job)
        if errors:
            messagebox.showerror("Validation error", "\n".join(errors))
            return
        idx = self._tree.index(sel[0])
        job["status"] = self._jobs[idx]["status"]
        self._jobs[idx] = job
        self._refresh_tree()

    def _remove_selected_job(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        del self._jobs[idx]
        self._refresh_tree()

    def _move_job(self, direction):
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        new_idx = idx + direction
        if not (0 <= new_idx < len(self._jobs)):
            return
        self._jobs[idx], self._jobs[new_idx] = self._jobs[new_idx], self._jobs[idx]
        self._refresh_tree()
        children = self._tree.get_children()
        self._tree.selection_set(children[new_idx])

    def _refresh_tree(self):
        self._tree.delete(*self._tree.get_children())
        for job in self._jobs:
            name = job["outdir"] or "(no output dir)"
            self._tree.insert('', tk.END, values=(name, job["status"]))

    # --- run controls ---

    def _build_run_controls(self):
        row = tk.Frame(self)
        row.pack(fill='x', padx=10, pady=(0, 4))
        self._stop_on_error_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row, text="Stop queue on first error", variable=self._stop_on_error_var).pack(side='left')
        self._stop_btn = tk.Button(row, text="Stop After Current Job", command=self._on_stop_queue)
        self._stop_btn.pack(side='right', padx=(0, 8))
        self._run_btn = tk.Button(row, text="Run Queue", font=('TkDefaultFont', 11, 'bold'), command=self._on_run_queue)
        self._run_btn.pack(side='right')

    def _on_run_queue(self):
        if not self._jobs:
            messagebox.showerror("Empty queue", "Add at least one job to the queue first.")
            return
        self._run_btn.config(state='disabled')
        self._stop_requested = False
        threading.Thread(target=self._run_queue, daemon=True).start()

    def _on_stop_queue(self):
        self._stop_requested = True
        self._log.write("\n--- Stop requested: will halt after the current job finishes ---\n")

    def _run_queue(self):
        script = str(Path(__file__).parent.parent / "calvo_independent_train.py")
        train_python = resolve_training_python()
        self._log.write(f"Using training Python: {train_python}\n")
        total = len(self._jobs)
        for i, job in enumerate(self._jobs):
            if self._stop_requested:
                self._log.write("\n--- Queue stopped by user ---\n")
                break
            name = job["outdir"] or f"job {i + 1}"
            self._set_job_status(i, "Running")
            self._log.write(f"\n--- [{i + 1}/{total}] Starting: {name} ---\n")

            cmd = build_train_cmd(job, script, train_python)
            status = "Done"
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in iter(proc.stdout.readline, ''):
                    self._log.write(f"[{i + 1}/{total}] {line}")
                proc.stdout.close()
                proc.wait()
                if proc.returncode != 0:
                    status = f"Error (exit {proc.returncode})"
            except Exception as exc:
                status = f"Error: {exc}"

            self._set_job_status(i, status)
            self._log.write(f"--- [{i + 1}/{total}] {name}: {status} ---\n")

            if status != "Done" and self._stop_on_error_var.get():
                self._log.write("\n--- Stopping queue after error ---\n")
                break

        self.after(0, self._run_btn.config, {'state': 'normal'})

    def _set_job_status(self, idx, status):
        self._jobs[idx]["status"] = status

        def update():
            children = self._tree.get_children()
            if idx < len(children):
                values = list(self._tree.item(children[idx], 'values'))
                values[1] = status
                self._tree.item(children[idx], values=values)
        self.after(0, update)


if __name__ == '__main__':
    app = BatchTrainingGUI()
    app.mainloop()
