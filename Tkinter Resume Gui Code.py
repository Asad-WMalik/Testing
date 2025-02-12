import tkinter as tk
from tkinter import filedialog, messagebox
from sentence_transformers import SentenceTransformer, util
import os

class ResumeRankerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Ranker using MiniLM-l6-v2")
        self.root.geometry("400x250")  # Set window size
        # Sentence Transformer model
        #self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = SentenceTransformer('Nashhz/SBERT_KFOLD_Job_Descriptions_to_Skills')
        # Widgets for the GUI
        tk.Label(root, text="Enter Job Description:", width=30, height=2).pack(pady=5)

        self.job_description_text = tk.Text(root, height=2, width=30)
        self.job_description_text.pack(pady=5)

        tk.Button(root, text="Upload Resumes", width=12, height=1, command=self.upload_resumes).pack(pady=5)

        tk.Button(root, text="Rank Resumes", width=12, height=1, command=self.rank_resumes).pack(pady=5)

        self.result_text = tk.Text(root, height=10, width=30, state=tk.DISABLED)
        self.result_text.pack(pady=5)

        self.resumes = []

    def upload_resumes(self):
        file_paths = filedialog.askopenfilenames(title="Select Resumes", filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))
        if file_paths:
            self.resumes = []
            for file_path in file_paths:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        self.resumes.append((os.path.basename(file_path), file.read()))
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Error reading {file_path}: {str(e)}")
            
            messagebox.showinfo("Success", "Resumes uploaded successfully!")

    def rank_resumes(self):
        job_description = self.job_description_text.get("1.0", tk.END).strip()
        if not job_description:
            messagebox.showwarning("Input Error", "Please enter a job description.")
            return

        if not self.resumes:
            messagebox.showwarning("Input Error", "Please upload at least one resume.")
            return

        # Encode job description and resumes
        job_embedding = self.model.encode(job_description, convert_to_tensor=True)
        resume_scores = []

        for name, content in self.resumes:
            resume_embedding = self.model.encode(content, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(job_embedding, resume_embedding).item()
            resume_scores.append((name, similarity))

        # Sort resumes by similarity score
        resume_scores.sort(key=lambda x: x[1], reverse=True)

        # Display results
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Ranked Resumes:\n")
        for rank, (name, score) in enumerate(resume_scores, start=1):
            self.result_text.insert(tk.END, f"{rank}. {name} - Score: {score:.4f}\n")
        self.result_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeRankerApp(root)
    root.mainloop()
