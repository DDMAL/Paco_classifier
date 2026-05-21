import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading, queue, os, sys
from itertools import product
import cv2, numpy as np
from Paco_classifier import recognition_engine as recognition

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


class BatchClassifierGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Paco Batch Classifier")
        self._log_queue = queue.Queue()
        self.build_inputs()
        self.build_bg_models()
        self.build_layer_models()
        self.build_params()
        self.build_run_button()
        self.build_output_area()
        self.poll_log_queue()

    # Building GUI layout

    def build_inputs(self):
        frame = tk.LabelFrame(self, text="Inputs", padx=8, pady=8)
        frame.pack(fill='x', padx=10, pady=5)

        self.image_dir_var = tk.StringVar()
        self.make_browse_row(frame, "Image Folder:", self.image_dir_var,
                             lambda: filedialog.askdirectory())

        self.outdir_var = tk.StringVar()
        self.make_browse_row(frame, "Output Dir:", self.outdir_var,
                             lambda: filedialog.askdirectory())

    def build_bg_models(self):
        frame = tk.LabelFrame(self, text="Background Models", padx=8, pady=8)
        frame.pack(fill='x', padx=10, pady=5)

        self.bg_frame = tk.Frame(frame)
        self.bg_frame.pack(anchor='w')
        self.bg_vars = []
        self.bg_rows = []
        self.add_model_row(self.bg_frame, self.bg_vars, self.bg_rows)

        btn_row = tk.Frame(frame)
        btn_row.pack(anchor='w', pady=(2, 0))
        tk.Button(btn_row, text="+ Add Background Model",
                  command=lambda: self.add_model_row(self.bg_frame, self.bg_vars, self.bg_rows)).pack(side='left')

    def build_layer_models(self):
        frame = tk.LabelFrame(self, text="Layer Models", padx=8, pady=8)
        frame.pack(fill='x', padx=10, pady=5)

        self.layer_frame = tk.Frame(frame)
        self.layer_frame.pack(anchor='w')
        self.layer_vars = []
        self.layer_rows = []
        self.add_model_row(self.layer_frame, self.layer_vars, self.layer_rows)

        btn_row = tk.Frame(frame)
        btn_row.pack(anchor='w', pady=(2, 0))
        tk.Button(btn_row, text="+ Add Layer Model",
                  command=lambda: self.add_model_row(self.layer_frame, self.layer_vars, self.layer_rows)).pack(side='left')

    def build_params(self):
        frame = tk.LabelFrame(self, text="Parameters", padx=8, pady=8)
        frame.pack(fill='x', padx=10, pady=5)

        spin_row = tk.Frame(frame)
        spin_row.pack(anchor='w')
        for label, attr, default in [("Height", "height_var", 256),
                                      ("Width",  "width_var",  256)]:
            tk.Label(spin_row, text=label).pack(side='left')
            var = tk.StringVar(value=str(default))
            setattr(self, attr, var)
            ttk.Spinbox(spin_row, from_=1, to=9999, textvariable=var,
                        width=6).pack(side='left', padx=(0, 12))

    def build_run_button(self):
        self.run_btn = tk.Button(self, text="Run Batch",
                                 command=self.on_run, font=('TkDefaultFont', 11, 'bold'))
        self.run_btn.pack(pady=8)

    def build_output_area(self):
        log_frame = tk.LabelFrame(self, text="Progress", padx=8, pady=8)
        log_frame.pack(fill='x', padx=10, pady=5)

        self.status_label = tk.Label(log_frame, text="", anchor='w')
        self.status_label.pack(fill='x')

        self.log = scrolledtext.ScrolledText(log_frame, height=10, state='disabled',
                                             font=('TkFixedFont',))
        self.log.pack(fill='x')

    # Helper methods

    def make_browse_row(self, parent, label, var, cmd):
        row = tk.Frame(parent)
        row.pack(anchor='w', pady=2)
        tk.Label(row, text=label, width=18, anchor='w').pack(side='left')
        tk.Entry(row, textvariable=var, width=50).pack(side='left')
        tk.Button(row, text="Browse",
                  command=lambda: var.set(cmd())).pack(side='left')

    def add_model_row(self, parent_frame, vars_list, rows_list, path=""):
        var = tk.StringVar(value=path)
        row = tk.Frame(parent_frame)
        tk.Entry(row, textvariable=var, width=50).pack(side='left')
        tk.Button(row, text="Browse",
                  command=lambda v=var: v.set(filedialog.askopenfilename())).pack(side='left')
        tk.Button(row, text="x",
                  command=lambda r=row, v=var, vl=vars_list, rl=rows_list: self.remove_model_row(r, v, vl, rl)).pack(side='left')
        row.pack(anchor='w')
        vars_list.append(var)
        rows_list.append(row)

    def remove_model_row(self, row, var, vars_list, rows_list):
        if len(vars_list) <= 1:
            return
        row.destroy()
        vars_list.remove(var)
        rows_list.remove(row)

    def on_run(self):
        self.run_btn.config(state='disabled')
        self.log.config(state='normal')
        self.log.delete('1.0', tk.END)
        self.log.config(state='disabled')
        self.status_label.config(text="Starting...")
        threading.Thread(target=self.run_batch, daemon=True).start()

    def run_batch(self):
        old_stdout = sys.stdout
        sys.stdout = _StdoutRedirector(self._log_queue)
        try:
            image_dir = self.image_dir_var.get()
            output_dir = self.outdir_var.get()
            height = int(self.height_var.get())
            width = int(self.width_var.get())
            bg_paths = [v.get() for v in self.bg_vars if v.get()]
            layer_paths = [v.get() for v in self.layer_vars if v.get()]

            if not image_dir or not output_dir:
                print("Error: image folder and output dir are required.")
                return
            if not bg_paths:
                print("Error: at least one background model is required.")
                return
            if not layer_paths:
                print("Error: at least one layer model is required.")
                return

            image_files = sorted(
                f for f in os.listdir(image_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            )
            if not image_files:
                print(f"Error: no images found in {image_dir!r}")
                return

            combos = list(product(bg_paths, layer_paths))
            total_combos = len(combos)
            total_images = len(image_files)
            print(f"Found {total_images} image(s), {total_combos} model combination(s) "
                  f"-> {total_combos * total_images} total runs\n")

            for combo_idx, (bg_path, layer_path) in enumerate(combos):
                bg_stem = os.path.splitext(os.path.basename(bg_path))[0]
                layer_stem = os.path.splitext(os.path.basename(layer_path))[0]
                combo_dir = os.path.join(output_dir, f"{bg_stem}__{layer_stem}")
                os.makedirs(combo_dir, exist_ok=True)
                model_paths = [bg_path, layer_path]

                for img_idx, img_file in enumerate(image_files):
                    status = (f"Combo {combo_idx + 1}/{total_combos} "
                              f"— image {img_idx + 1}/{total_images}: {img_file}")
                    self.after(0, self.status_label.config, {'text': status})
                    print(status)

                    img_path = os.path.join(image_dir, img_file)
                    image = cv2.imread(img_path, 1)
                    if image is None:
                        print(f"  Warning: could not read {img_path!r}, skipping")
                        continue

                    image_base = os.path.splitext(img_file)[0]
                    analyses = recognition.process_image_msae(
                        image, model_paths, height, width, mode='logical')

                    for id_label, _ in enumerate(model_paths):
                        label_range = np.array(id_label, dtype=np.uint8)
                        mask = cv2.inRange(analyses, label_range, label_range)
                        masked = cv2.bitwise_and(image, image, mask=mask)
                        masked[mask == 0] = (255, 255, 255)
                        alpha = np.ones(mask.shape, dtype=mask.dtype) * 255
                        alpha[mask == 0] = 0
                        b, g, r = cv2.split(masked)
                        rgba = cv2.merge((b, g, r, alpha))
                        fname = (f'background_layer_{image_base}.png' if id_label == 0
                                 else f'layer_{id_label}_{image_base}.png')
                        cv2.imwrite(os.path.join(combo_dir, fname), rgba)

            print("\nBatch complete.")
            self.after(0, self.status_label.config, {'text': "Done."})
        finally:
            sys.stdout = old_stdout
            self.after(0, lambda: self.run_btn.config(state='normal'))

    def poll_log_queue(self):
        while True:
            try:
                msg = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self.log.config(state='normal')
            self.log.insert(tk.END, msg)
            self.log.see(tk.END)
            self.log.config(state='disabled')
        self.after(100, self.poll_log_queue)


class _StdoutRedirector:
    def __init__(self, q):
        self.q = q

    def write(self, msg):
        self.q.put(msg)

    def flush(self):
        pass


if __name__ == '__main__':
    app = BatchClassifierGUI()
    app.mainloop()
