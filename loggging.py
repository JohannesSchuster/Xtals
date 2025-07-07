import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

class Logging:
    _instance = None

    def __init__(self, parent=None):
        if Logging._instance is not None:
            raise Exception("Use Logging.get() to access the logger.")
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title('Log')
        self._build_menu()
        self._build_widgets()
        self.log_lines = []
        Logging._instance = self

    @classmethod
    def get(cls, parent=None):
        if cls._instance is None:
            cls._instance = Logging(parent)
        return cls._instance

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save Log", command=self.save_log)
        filemenu.add_command(label="Load Log", command=self.load_log)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

    def _build_widgets(self):
        frame = ttk.Frame(self.root)
        frame.pack(fill='both', expand=True)
        self.text = tk.Text(frame, wrap='word', state='disabled', height=20, width=80)
        self.text.pack(fill='both', expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(frame, command=self.text.yview)
        self.text['yscrollcommand'] = scrollbar.set
        scrollbar.pack(side='right', fill='y')

    def log(self, message: str):
        self.log_lines.append(message)
        self.text.config(state='normal')
        self.text.insert('end', message + '\n')
        self.text.see('end')
        self.text.config(state='disabled')

    @staticmethod
    def log(message: str):
        if Logging._instance is not None:
            Logging._instance.log(message)
        else:
            print(f"Logging instance not initialized. Message: {message}")
        
    def save_log(self):
        file_path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text Files', '*.txt')])
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(self.log_lines))
            except Exception as e:
                messagebox.showerror('Error', f'Could not save log: {e}')

    def load_log(self):
        file_path = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
        if file_path and os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                self.log_lines = lines
                self.text.config(state='normal')
                self.text.delete('1.0', 'end')
                for line in lines:
                    self.text.insert('end', line + '\n')
                self.text.config(state='disabled')
            except Exception as e:
                messagebox.showerror('Error', f'Could not load log: {e}')

if __name__ == '__main__':
    logwin = Logging.get()
    logwin.log('Logging window started.')
    logwin.root.mainloop()
