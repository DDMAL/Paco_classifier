import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk


class TrainingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calvo Independent Training")
        self.resizable(True, True)

        self._image_rows = []
        self._bg_rows = []
        self._layer_rows = []
        self._region_rows = []

        self._log_queue = queue.Queue()

        self._build_images()
        self._build_bg_masks()
        self._build_layer_masks()
        self._build_region_masks()
        self._build_params()
        self._build_output()
        self._build_run()
        self._build_log()

        self._add_row(self._image_rows, self._images_frame, "Image")
        self._add_row(self._bg_rows, self._bg_frame, "BG Mask")
        self._add_row(self._layer_rows, self._layer_frame, "Layer Mask")

        self.after(100, self._poll_log)

    # build

    def _build_images(self):
        self._images_frame = self._section("Input Images")
        tk.Button(self._images_frame, text="+ Add Image",
                  command=lambda: self._add_row(self._image_rows, self._images_frame, "Image")
                  ).pack(anchor='w', pady=(4, 0))
    
    def _build_bg_masks(self):
        self._bg_frame = self._section("Backgroun Masks (one per image, RGBA PNG)")
        tk.Button(self._bg_frame, text="+ Add BG Mask",
                  command=lambda: self._add_row(self._bg_rows, self._bg_frame, "BG Mask")
                  ).pack(anchor='w', pady=(4, 0))
        
    def _build_layer_masks(self):
        self._layer_frame = self._section("Layer Masks (flat list, N-images x L-layers, RGBA PNG)")
        tk.Button(self._layer_frame, text="+ Add Layer Mask",
                  command=lambda: self._add_row(self._layer_rows, self._layer_frame, "Layer Mask")
                  ).pack(anchor='w', pady=(4, 0))
        
    def _build_region_masks(self):
        self._region_frame = self._section("Regions Masks (optional, one per image, RGBA PNG)")
        tk.Button(self._region_frame, text="+ Add Region Mask",
                  command=lambda: self._add_row(self._region_rows, self._region_frame, "Region Mask")
                  ).pack(anchor='w', pady=(4, 0))
        
    def _build_params(self):
        frame = self._section("Parameters")
        grid = tk.Frame(frame)
        grid.pack(anchor='w')

        self._height_var = self._spinbox_row(grid, 0, "Height", 256, 1, 9999)
        self._width_var = self._spinbox_row(grid, 1, "Width", 256, 1, 9999)
        self._epochs_var = self._spinbox_row(grid, 2, "Epochs", 50, 1, 9999)
        self._maxsamp_var = self._spinbox_row(grid, 3, "Max Samples/Class", 1000, 1, 999999)
        self._batch_var = self._spinbox_row(grid, 4, "Batch Size", 16, 1, 4096)
        self._ram_var = self._spinbox_row(grid, 5, "RAM Limit (GB)", 4.0, 0.1, 999.0, inc=0.5)

    def _build_output(self):
        frame = self._section("Output Directory")
        self._outdir_var = tk.StringVar()
        self._make_browse_row(frame, "Output dir:", self._outdir_var, filedialog.askdirectory)

    def _build_run(self):
        self._run_btn = tk.Button(self, text="Run Training",
                                  font=('TkDefaultFont', 11, 'bold'),
                                  command=self._on_run)
        self._run_btn.pack(pady=8)

    def _build_log(self):
        frame = tk.LabelFrame(self, text="Log", padx=8, pady=8)
        frame.pack(fill='both', expand=True, padx=10, pady=5)
        self._log = scrolledtext.ScrolledText(frame, height=14, state='disabled',
                                              wrap='word', font=('Courier', 10))
        self._log.pack(fill='both', expand=True)

    
    # helpers
    def _section(self, title):
        f = tk.LabelFrame(self, text=title, padx=8, pady=8)
        f.pack(fill='x', padx=10, pady=4)
        return f
    
    def _make_browse_row(self, parent, label, var, cmd):
        row = tk.Frame(parent)
        row.pack(anchor='w', pady=2)
        tk.Label(row, text=label, width=20, anchor='w').pack(side='left')
        tk.Entry(row, textvariable=var, width=50).pack(side='left')
        tk.Button(row, text="Browse", command=lambda: var.set(cmd())).pack(side='left')

    def _spinbox_row(self, grid, r, label, default, lo, hi, inc=1.0):
        var = tk.StringVar(value=str(default))
        tk.Label(grid, text=label + ":", anchor='w', width=22).grid(row=r, column=0, sticky='w', pady=2)
        ttk.Spinbox(grid, textvariable=var, from_=lo, to=hi, increment=inc, width=10).grid(row=r, column=1, sticky='w')
        return var
    
    def _add_row(self, rows_list, frame, label_prefix):
        idx = len(rows_list) + 1
        var = tk.StringVar()

        # insert before the last child (the "+ Add" button)
        row = tk.Frame(frame)
        children = frame.pack_slaves()
        if children:
            row.pack(anchor='w', pady=2, before=children[-1])
        else:
            row.pack(anchor='w', pady=2)

        tk.Label(row, text=f"{label_prefix} {idx}:", width=14, anchor='w').pack(side='left')
        tk.Entry(row, textvariable=var, width=48).pack(side='left', padx=(0, 2))
        tk.Button(row, text="Browse",
                  command=lambda v=var: v.set(
                      filedialog.askopenfilename(
                          filetypes=[("Image / PNG", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                                     ("All files", "*.*")])
                  )).pack(side='left')
        tk.Button(row, text="x",
                  command=lambda r=row, v=var: self._remove_row(rows_list, r, v)
                  ).pack(side='left', padx=(2, 0))
        
        rows_list.append((row, var))
    
    def _remove_row(self, rows_list, row_frame, var):
        entry = (row_frame, var)
        if entry in rows_list:
            rows_list.remove(entry)
        row_frame.destroy()

    # run

    def _on_run(self):
        images = [v.get().strip() for _, v in self._image_rows if v.get().strip()]
        bg = [v.get().strip() for _, v in self._bg_rows if v.get().strip()]
        layers = [v.get().strip() for _, v in self._layer_rows if v.get().strip()]
        regions = [v.get().strip() for _, v in self._region_rows if v.get().strip()]
        outdir = self._outdir_var.get().strip()

        # validation
        errors = []
        if not images:
            errors.append("Add at least one image.")
        if len(bg) != len(images):
            errors.append(f"Background masks ({len(bg)}) must equal image count ({len(images)}).")
        if not layers or len(layers) % max(len(images), 1) != 0:
            errors.append(f"Layer masks ({len(layers)}) must be a non-zero multiple of image count ({len(images)}).")
        if regions and len(regions) != len(images):
            errors.append(f"Region masks ({len(regions)}) must equal image count ({len(images)}) or be empty.")
        if not outdir:
            errors.append("Choose an output directory.")
        if errors:
            messagebox.showerror("Validation error", "\n".join(errors))
            return
        
        self._run_btn.config(state='disabled')
        self._log_write("\n--- Starting training ---\n")
        threading.Thread(target = self._run_training,
                         args=(images, bg, layers, regions, outdir),
                         daemon=True).start()
        
    def _run_training(self, images, bg, layers, regions, outdir):
        script = str(Path(__file__).parent.parent / "calvo_independent_train.py")
        cmd = [
            sys.executable, script,
            "--images", *images,
            "--background-mask", *bg,
            "--layer-masks", *layers,
            "--output-dir", outdir,
            "--height", self._height_var.get(),
            "--width", self._width_var.get(),
            "--epochs", self._epochs_var.get(),
            "--max-samples-per-class", self._maxsamp_var.get(),
            "--batch-size", self._batch_var.get(),
            "--ram-limit", self._ram_var.get(),
        ]
        if regions:
            cmd == ["--regions-mask", *regions]

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(proc.stdout.readline, ''):
                self._log_queue.put(line)
            proc.stdout.close()
            proc.wait()
        except Exception as exc:
            status = f"Error: {exc}"

        self._log_queue.put(f"\n--- {status} ---\n")
        self.after(0, self._run_btn.config, {'state': 'normal'})

    # log

    def _log_write(self, text):
        self._log.config(state='normal')
        self._log.insert(tk.END, text)
        self._log.see(tk.END)
        self._log.config(state='disabled')

    def _poll_log(self):
        while True:
            try:
                msg = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._log_write(msg)
        self.after(100, self._poll_log)


if __name__ == '__main__':
    app = TrainingGUI()
    app.mainloop()


