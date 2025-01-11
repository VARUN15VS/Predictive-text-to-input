import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset: Text corpus
text_corpus = """Soft computing embraces approximate solutions to complex problems.
Predictive text systems utilize RNNs for handling sequential data efficiently.
Neuro-fuzzy systems improve adaptability in linguistic tasks."""

# Parameters
vocab_size = 5000  # Maximum vocabulary size
embedding_dim = 50  # Dimension of embedding layer
max_sequence_len = 10  # Maximum sequence length
rnn_units = 128  # Number of RNN units

def preprocess_data(corpus):
    """Preprocesses the text corpus for model training."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts([corpus])
    total_words = len(tokenizer.word_index) + 1

    # Create input sequences
    input_sequences = []
    for line in corpus.split("\n"):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences to ensure consistent input size
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

    # Split into predictors and labels
    predictors, labels = input_sequences[:,:-1], input_sequences[:,-1]
    labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)
    return predictors, labels, total_words, tokenizer

# Preprocess data
predictors, labels, total_words, tokenizer = preprocess_data(text_corpus)

# Build the RNN model
model = Sequential([
    Embedding(total_words, embedding_dim, input_length=max_sequence_len-1),
    SimpleRNN(rnn_units, return_sequences=False),
    Dense(total_words, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 100
model.fit(predictors, labels, epochs=epochs, verbose=1)

# Function for generating text
def generate_text(seed_text, next_words, max_sequence_len):
    """Generates text based on a seed text and trained model."""
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Test the model
seed_text = "Predictive text"
generated_text = generate_text(seed_text, next_words=10, max_sequence_len=max_sequence_len)
print("Generated Text:", generated_text)
