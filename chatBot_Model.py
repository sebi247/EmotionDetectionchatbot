import numpy as np
import random
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D ,TimeDistributed, Activation
from keras.regularizers import L1L2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
import csv
from keras.optimizers import Adam
import gensim.downloader as api
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score


class SaveValAccuracyLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        super(SaveValAccuracyLossCallback, self).__init__()
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_accuracy = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        with open(self.filename, 'a') as f:
            f.write(f'{val_accuracy},{val_loss}\n')



embedding_model = api.load('glove-wiki-gigaword-100')


with open('dataset_Final.csv', 'r') as file:
    reader = csv.reader(file)
    dataset = list(reader)

texts = [entry[0] for entry in dataset if len(entry) > 1]
labels = [entry[1].strip() for entry in dataset if len(entry) > 1]

label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post')

tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))


vocab_size = len(tokenizer.word_index) + 1
encoded_labels = [label_to_id[label] for label in labels]

embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in tokenizer.word_index.items():
    if word in embedding_model.key_to_index:
        embedding_matrix[idx] = embedding_model[word]

X_train, X_val, y_train, y_val = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

y_train = np.array(y_train)
y_val = np.array(y_val)

lstm_units = 48
dropout_rate = 0.35
regularization = L1L2(l1=0.0001, l2=0.0001)
patience = 2
batch_size = 32
learning_rate = 0.001

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=padded_sequences.shape[1],
              weights=[embedding_matrix], trainable=True,
              embeddings_regularizer=regularization),
    Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, kernel_regularizer=regularization)),
    BatchNormalization(),
    Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, kernel_regularizer=regularization)),
    BatchNormalization(),
    TimeDistributed(Dense(lstm_units, activation='tanh')),
    GlobalMaxPooling1D(),
    Dropout(dropout_rate),
    Dense(len(label_to_id), activation='softmax')
])

input_length = padded_sequences.shape[1]

with open("input_length.txt", "w") as f:
    f.write(str(input_length))

optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_checkpoint = ModelCheckpoint('Chat_bot.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100000)

save_val_accuracy_loss = SaveValAccuracyLossCallback('val_accuracy_loss.csv')

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=200,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, model_checkpoint, reduce_lr, save_val_accuracy_loss])

model.save('Chat_bot_final.h5')

# Evaluate the model
train_loss, train_accuracy = model.evaluate(X_train, y_train)
val_loss, val_accuracy = model.evaluate(X_val, y_val)

print(f'Training loss: {train_loss:.4f}, Training accuracy: {train_accuracy:.4f}')
print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}')

y_train_pred = np.argmax(model.predict(X_train), axis=-1)
y_val_pred = np.argmax(model.predict(X_val), axis=-1)

train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1_score = f1_score(y_train, y_train_pred, average='weighted')

val_precision = precision_score(y_val, y_val_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')
val_f1_score = f1_score(y_val, y_val_pred, average='weighted')

print(f'Training precision: {train_precision:.4f}, Training recall: {train_recall:.4f}, Training F1-score: {train_f1_score:.4f}')
print(f'Validation precision: {val_precision:.4f}, Validation recall: {val_recall:.4f}, Validation F1-score: {val_f1_score:.4f}')