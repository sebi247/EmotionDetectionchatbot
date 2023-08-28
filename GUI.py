from cmath import e
import tkinter as tk
from tkinter import ttk
import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import mysql.connector
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import os
import textwrap

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, 'data', relative_path)
    return os.path.join(os.path.abspath('.'), 'data', relative_path)

class CustomListbox(tk.Listbox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind('<FocusOut>', self.clear_selection)
        self.wrap_width = 20  

    def clear_selection(self, event):
        self.selection_clear(0, tk.END)

    def insert_wrapped_message(self, message):
        wrapped_message = textwrap.fill(message, width=self.wrap_width)
        wrapped_lines = wrapped_message.split('\n')
        for line in wrapped_lines:
            self.insert(tk.END, line)


class ChatbotGUI:
    def __init__(self, master, root):
        with open(resource_path("input_length.txt"), "r") as f:
            self.input_length = int(f.read())

        self.pattern_intents = json.loads(open(resource_path('PatternIntents.json')).read())
        self.intents = json.loads(open(resource_path('Intents.json')).read())      
        self.words = pickle.load(open(resource_path('words.pkl'), 'rb'))       
        self.classes = pickle.load(open(resource_path('classes.pkl'), 'rb'))        
        self.emotion_model = load_model(resource_path('chatbot_answers.model'))   
        self.model = load_model(resource_path('Chat_bot.h5'))      
        self.tokenizer = self.init_tokenizer()

        self.master = master
        self.master.title("Sentiment Chatbot")
        self.master.config(bg="#282C34")

        labels = ['joy', 'disgust', 'anger', 'sadness', 'fear', 'surprise']

        self.init_database()

        self.conversation_frame = tk.Frame(self.master, bg="#282C34")
        self.conversation_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.conversation_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.conversation = CustomListbox(self.conversation_frame, yscrollcommand=self.scrollbar.set, width=80, height=20, bg="#3C3F58", fg="#ABB2BF", font=("Arial", 12))
        self.conversation.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar.config(command=self.conversation.yview)

        self.entry_frame = tk.Frame(self.master, bg="#282C34")
        self.entry_frame.pack(padx=10, pady=10, fill=tk.X, expand=True)

        self.message_entry = tk.Entry(self.entry_frame, width=60, font=("Arial", 12))
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.message_entry.bind('<Return>', self.send_message_wrapper)

        self.send_button = tk.Button(self.entry_frame, text="Send", command=self.send_message, bg="#61AFEF", fg="#282C34", font=("Arial", 12))
        self.send_button.pack(side=tk.RIGHT)

        logo_icon_path = resource_path("logo.ico")
        root.iconbitmap(logo_icon_path) 

        self.label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}      

        self.add_starting_message()



    def add_starting_message(self):
        starting_message = "Hello! I am an emotion-detecting chatbot. Type anything and I will try my best to label the emotion."
        self.conversation.insert(tk.END, "ChatBot: " + starting_message)
        self.conversation.see(tk.END)


    def init_tokenizer(self):
        # Load the saved tokenizer vocabulary
        with open(resource_path('tokenizer.json'), 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
        tokenizer_obj = json.loads(tokenizer_json)
        tokenizer = tokenizer_from_json(tokenizer_obj)
        return tokenizer


    def send_message_wrapper(self, event=None):
        self.send_message()

    def init_database(self):
        self.connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Sebires123",
            database="chatbot"
        )
        self.cursor = self.connection.cursor()

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question TEXT NOT NULL,
            emotion TEXT NOT NULL
        )
        """)


    def save_message_pair(self, question, emotion):
        self.cursor.execute(
            "INSERT INTO chat_data (question, emotion) VALUES (%s, %s)",
            (question, emotion)
        )
        self.connection.commit()


    def lemmatize_word(self, word):
        lemmatizer = WordNetLemmatizer()
        return lemmatizer.lemmatize(word)

    def send_message(self):
        user_message = self.message_entry.get().strip()
        if user_message:
            self.conversation.insert(tk.END, f"You: {user_message}")
            self.message_entry.delete(0, tk.END)
            self.master.update()

        # First, try to get a pattern-based response
        pattern_response = self.get_pattern_response(user_message)

        if pattern_response:
            response = pattern_response
        else:
        # If no pattern found, get the emotion-based response
            response = self.get_emotion_based_response(user_message)

        self.conversation.insert(tk.END, f"ChatBot: {response}")



    def get_pattern_response(self, message):
        message = [self.lemmatize_word(word.lower()) for word in nltk.word_tokenize(message)]

        # Search the pattern intents file first
        for intent in self.pattern_intents['Intents']:
            if self.message_match(intent['Patterns'], message):
                return random.choice(intent['responses'])

        return None



    def get_emotion_based_response(self, message):
        emotion = self.detect_emotion(message)
        response = self.get_response_by_emotion(emotion)
        self.save_message_pair(message, emotion) 
        return response



    def message_match(self, patterns, message):
        for pattern in patterns:
            pattern = [self.lemmatize_word(word.lower()) for word in nltk.word_tokenize(pattern)]
            if all(word in message for word in pattern):
                return True
        return False

    def get_response(self, message):
        pattern_response = self.get_pattern_response(message)
        if pattern_response:
            return pattern_response
        else:
            emotion = self.detect_emotion(message)
            response = self.get_response_by_emotion(emotion)
            return response


    def get_response_by_emotion(self, emotion):
        for intent in self.intents['Intents']:
            if intent['tag'] == emotion:
                return random.choice(intent['responses'])
        return "I am not sure how to respond to that."

    def detect_emotion(self, user_input):
        processed_input = self.preprocess_input(user_input)
        predicted_label = np.argmax(self.model.predict(processed_input), axis=-1)
        emotion = self.get_emotion(predicted_label[0])
        return emotion

    def preprocess_input(self, user_input):
        sequences = self.tokenizer.texts_to_sequences([user_input])
        padded_sequences = pad_sequences(sequences, padding='post', maxlen=self.input_length)
        return padded_sequences

    def get_emotion(self, label_id):
        id_to_label = {v: k for k, v in self.label_to_id.items()}
        print(f"Trying to get emotion for label_id: {label_id}")
        return id_to_label[label_id]


