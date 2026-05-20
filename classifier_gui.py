import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading, queue, os, sys
import cv2, numpy as np
from Paco_classifier import recognition_engine as recognition

class ClassifierGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Paco Classifier")
        self._log_queue = queue.Queue()
        self.build_inputs()
        self.build_params()
        self.build_run_button()
        self.build_output_area()
        self.poll_log_queue()


    # Building GUI layout 

    def build_inputs(self):
          frame = tk.LabelFrame(self, text="Inputs", padx=8, pady=8)
          frame.pack(fill='x', padx=10, pady=5)

          #Image row
          self.image_var = tk.StringVar()
          self.make_browse_row(frame, "Image:", self.image_var, 
                                lambda: filedialog.askopenfilename())
          #Background model row 
          self.bg_model_var = tk.StringVar()
          self.make_browse_row(frame, "Background Model:", self.bg_model_var,
                                lambda: filedialog.askopenfilename())
          
          #Layer models - dynamic list (up to 10)
          tk.Label(frame, text="Layer Models:").pack(anchor='w')
          self.layer_frame = tk.Frame(frame)
          self.layer_frame.pack(anchor='w')
          self.layer_vars = []
          self.layer_rows = []
          self.add_layer_row()

          btn_row = tk.Frame(frame)
          btn_row.pack(anchor='w', pady=(2, 0))
          tk.Button(btn_row, text="+ Add Layer",
                    command=self.add_layer_row).pack(side='left')
        
    def make_browse_row(self, parent, label, var, cmd):
          row = tk.Frame(parent)
          row.pack(anchor='w', pady=2)
          tk.Label(row, text=label, width=18, anchor='w').pack(side='left')
          tk.Entry(row, textvariable=var, width=50).pack(side='left')
          tk.Button(row, text="Browse",
                    command=lambda: var.set(cmd())).pack(side='left')
          
    def build_params(self):
        frame = tk.LabelFrame(self, text="Parameters", padx=8, pady=8)
        frame.pack(fill='x', padx=10, pady=5)

        spin_row = tk.Frame(frame)
        spin_row.pack(anchor='w')
        for label, attr, default in [("Height", "height_var", 256),
                                       ("Width", "width_var", 256),
                                       ("Threshold", "thresh_var", 50)]:
            tk.Label(spin_row, text=label).pack(side='left')
            var = tk.StringVar(value=str(default))
            setattr(self, attr, var)
            ttk.Spinbox(spin_row, from_=1, to=9999, textvariable=var,
                        width=6).pack(side='left', padx=(0, 12))
                
        self.outdir_var = tk.StringVar()
        self.make_browse_row(frame, "Output dir:", self.outdir_var,
                                lambda: filedialog.askdirectory())
        

    def build_run_button(self):
        self.run_btn = tk.Button(self, text="Run Classification",
                                 command=self.on_run, font=('TkDefaultFont', 11, 'bold'))
        self.run_btn.pack(pady=8)

    def build_output_area(self):
         log_frame = tk.LabelFrame(self, text="progress", padx=8, pady=8)
         log_frame.pack(fill='x', padx=10, pady=5)
         self.log = scrolledtext.ScrolledText(log_frame, height=8, state='disabled',
                                              font=('TkFixedFont',))
         
         self.log.pack(fill='x')

         thumb_outer = tk.LabelFrame(self, text="Output Images", padx=8, pady=8)
         thumb_outer.pack(fill='x', padx=10, pady=5)
         self.thumb_frame = tk.Frame(thumb_outer)
         self.thumb_frame.pack(anchor='w')
         self._thumb_refs = []

    # Helper methods

    def add_layer_row(self, path=""):
         var = tk.StringVar(value=path)
         row = tk.Frame(self.layer_frame)
         tk.Entry(row, textvariable=var, width=50).pack(side='left')
         tk.Button(row, text="Browse",
                   command=lambda v=var: v.set(filedialog.askopenfilename())).pack(side='left')
         tk.Button(row, text="x", command=lambda r=row, v=var: self.remove_layer_row(r, v)).pack(side='left')
         row.pack(anchor='w')
         self.layer_vars.append(var)
         self.layer_rows.append(row)

    def remove_layer_row(self, row, var):
         if len(self.layer_vars) <= 1:
              return
         row.destroy()
         self.layer_vars.remove(var)
         self.layer_rows.remove(row)

    def on_run(self):
         self.run_btn.config(state='disabled')
         self.log.config(state='normal')
         self.log.delete('1.0', tk.END)
         self.log.config(state='disabled')
         self.clear_thumbnails()
         threading.Thread(target=self.run_inference, daemon=True).start()

    def run_inference(self):
        old_stdout = sys.stdout
        sys.stdout = _StdoutRedirector(self._log_queue)
        try:
            image = cv2.imread(self.image_var.get(), 1)
            if image is None:
                print(f"Error: cannot read {self.image_var.get()!r}")
                return
            output_dir = self.outdir_var.get() or os.path.dirname(os.path.abspath(self.image_var.get()))
            os.makedirs(output_dir, exist_ok=True)
            model_paths = [self.bg_model_var.get()] + [v.get() for v in self.layer_vars if v.get()]
            height = int(self.height_var.get())
            width = int(self.width_var.get())
            analyses = recognition.process_image_msae(image, model_paths, height, width, mode='logical')
            output_paths = []
            for id_label, _ in enumerate(model_paths):
                   label_range = np.array(id_label, dtype=np.uint8)
                   mask = cv2.inRange(analyses, label_range, label_range)
                   masked = cv2.bitwise_and(image, image, mask=mask)
                   masked[mask == 0] = (255, 255, 255)
                   alpha = np.ones(mask.shape, dtype=mask.dtype) * 255
                   alpha[mask == 0] = 0
                   b, g, r = cv2.split(masked)
                   rgba = cv2.merge((b, g, r, alpha))
                   fname = 'background_layer.png' if id_label == 0 else f'layer_{id_label}.png'
                   path = os.path.join(output_dir, fname)
                   cv2.imwrite(path, rgba)
                   output_paths.append(path)
            self.after(0, self.show_results, output_paths)
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


    def show_results(self, output_paths):
         for path in output_paths:
              img = tk.PhotoImage(file=path)
              w = img.width()
              factor = max(1, w // 200) # subsample to fit ~200 px wide
              img = img.subsample(factor, factor)
              col = tk.Frame(self.thumb_frame)
              tk.Label(col, image=img).pack()
              tk.Label(col, text=os.path.basename(path), font=('TkSmallCaptionFont',)).pack()
              col.pack(side='left', padx=4)
              self._thumb_refs.append(img)
    
    def clear_thumbnails(self):
        for w in self.thumb_frame.winfo_children():
             w.destroy()
        self._thumb_refs.clear()


class _StdoutRedirector:
    def __init__(self, q):
         self.q = q

    def write(self, msg):
         self.q.put(msg)
    
    def flush(self):
        pass




if __name__ == '__main__':
    app = ClassifierGUI()
    app.mainloop()
