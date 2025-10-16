import os
from tkinter import *
from tkinter import filedialog, scrolledtext, messagebox
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Global variable
qa = None

def load_pdf():
    global qa
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if not file_path:
        return

    try:
        # Extract text from PDF
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        if not text.strip():
            messagebox.showerror("Error", "No text found in PDF.")
            return

        # Split text into chunks
        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_text(text)

        # Create embeddings and FAISS vector store
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = FAISS.from_texts(chunks, embeddings)
        retriever = db.as_retriever()

        # Load language model pipeline
        qa_model = pipeline("text2text-generation", model=LLM_MODEL, max_new_tokens=200)
        llm = HuggingFacePipeline(pipeline=qa_model)

        # Build RetrievalQA chain
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        messagebox.showinfo("Success", f"PDF loaded successfully:\n{os.path.basename(file_path)}")
        chat_area.insert(END, f"✅ PDF loaded: {os.path.basename(file_path)}\n\n")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load PDF:\n{str(e)}")

def ask_question():
    global qa
    if not qa:
        messagebox.showwarning("Warning", "Please load a PDF first!")
        return

    query = question_entry.get().strip()
    if not query:
        messagebox.showwarning("Warning", "Please enter a question!")
        return

    try:
        answer = qa.run(query)
        chat_area.insert(END, f"🧑 You: {query}\n🤖 Bot: {answer}\n\n")
        chat_area.see(END)
        question_entry.delete(0, END)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to answer the question:\n{str(e)}")

def clear_chat():
    chat_area.delete('1.0', END)

# GUI Setup
root = Tk()
root.title("AI PDF Chatbot")
root.geometry("650x600")

Label(root, text="AI PDF Chatbot (GPT-3.5 Powered)", font=("Arial", 18, "bold")).pack(pady=10)

# Load PDF button
Button(root, text="📂 Load PDF", command=load_pdf, width=20, font=("Arial", 12)).pack(pady=5)

# Chat area
chat_area = scrolledtext.ScrolledText(root, wrap=WORD, width=75, height=25, font=("Arial", 11))
chat_area.pack(pady=10)

# Question entry frame
question_frame = Frame(root)
question_frame.pack(pady=5)

question_entry = Entry(question_frame, width=50, font=("Arial", 12))
question_entry.pack(side=LEFT, padx=5)

Button(question_frame, text="Ask", command=ask_question, width=10, font=("Arial", 12)).pack(side=LEFT, padx=5)
Button(question_frame, text="Clear Chat", command=clear_chat, width=10, font=("Arial", 12)).pack(side=LEFT)

# Start GUI loop
root.mainloop()