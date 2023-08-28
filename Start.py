import tkinter as tk
from tkinter import ttk
from GUI import ChatbotGUI
import sys
import os

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, 'data', relative_path)
    return os.path.join(os.path.abspath('.'), 'data', relative_path)

class StartInterface:
    def __init__(self, master):
        self.master = master
        self.master.title("Sentiment Chatbot")
        self.master.geometry("500x600")
        self.master.config(bg="#282C34")

        # Create the logo frame and add the logo
        logo_frame = tk.Frame(self.master, width=500, height=350, bg="#282C34")
        logo_frame.pack_propagate(False)  # Prevent the frame from scaling to its content
        logo_frame.pack()

        logo_path = resource_path("logo.png")
        self.logo = tk.PhotoImage(file=logo_path)
        self.logo_label = tk.Label(logo_frame, image=self.logo, bg="#282C34")
        self.logo_label.place(relx=0.5, rely=0.5, anchor="center")

        # Create a label for the title
        title_label = tk.Label(self.master, text="Sentiment Chatbot", font=("Arial", 24, "bold"), bg="#282C34", fg="#61AFEF")
        title_label.pack(pady=10)

        # Create a label for a short description
        description_label = tk.Label(self.master, text="Welcome! Let's start a chat and analyze the sentiment.", wraplength=450, font=("Arial", 12), bg="#282C34", fg="#ABB2BF")
        description_label.pack(pady=10)

        # Create the start chat button
        start_button = ttk.Button(self.master, text="Start a New Chat", command=self.start_chat, style="Custom.TButton", takefocus=False)
        start_button.pack(pady=20)

        # Style the button
        style = ttk.Style()
        style.configure("Custom.TButton", font=("Arial", 18), background="#61AFEF", foreground="#282C34")
        style.map("Custom.TButton", background=[("active", "#3F7EBD"), ("disabled", "#4B5263")])

        self.chat_window = None

    def start_chat(self):
        self.master.withdraw()  # Hide the StartInterface window
        self.chat_window = tk.Toplevel()  # Use Toplevel to create a new window
        self.chat_window.title("Sentiment Chatbot")
        self.chat_window.protocol("WM_DELETE_WINDOW", self.close_chat)  # Handle the chat window being closed

        app = ChatbotGUI(self.chat_window, self.master)  # Pass the root variable to the ChatbotGUI constructor

        self.chat_window.deiconify()  # Show the chat window

    def close_chat(self):
        self.master.deiconify()  # Restore the StartInterface window
        self.chat_window.destroy()  # Close the ChatbotGUI window

root = tk.Tk()
app = StartInterface(root)
root.mainloop()
