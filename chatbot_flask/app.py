from flask import Flask, render_template, request
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spellchecker import SpellChecker
import os

app = Flask(__name__)

# Initialize an in-memory list to store chat history
chat_history = []

# Load spaCy's English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize the spell checker
spell = SpellChecker()

FILE_PATH = "documents/scraped_data1.csv"

def split_into_sentences(text):
    if isinstance(text, float):
        text = str(text) if not pd.isna(text) else ""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    print(f"Split Sentences: {sentences}")  # Debugging line
    return sentences

def lemmatize_sentence(sentence):
    doc = nlp(sentence.lower())
    lemmatized_words = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    lemmatized_sentence = ' '.join(lemmatized_words)
    print(f"Lemmatized Sentence: {lemmatized_sentence}")  # Debugging line
    return lemmatized_sentence

def correct_spelling(text):
    words = text.split()
    corrected_words = [spell.correction(word) or word for word in words]  # Use the original word if correction is None
    print(f"Corrected Words: {corrected_words}")  # Debugging line
    return ' '.join(corrected_words)

def find_relevant_sentences(query, sentences, lemmatized_sentences):
    lemmatized_query = lemmatize_sentence(query)
    print(f"Lemmatized Query: {lemmatized_query}")  # Debugging line

    # Fuzzy matching
    scores = [fuzz.partial_ratio(lemmatized_query, ls) for ls in lemmatized_sentences]
    best_matches = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    print(f"Fuzzy Matching Scores: {scores}")  # Debugging line

    threshold = 80  # Adjusted threshold
    relevant_sentences = [sentences[i] for i, score in best_matches if score >= threshold]
    
    if not relevant_sentences:
        # Use TF-IDF and cosine similarity
        vectorizer = TfidfVectorizer().fit(sentences + [lemmatized_query])
        vectors = vectorizer.transform(sentences + [lemmatized_query])
        cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
        print(f"Cosine Similarities: {cosine_similarities}")  # Debugging line

        best_idx = cosine_similarities.argmax()
        if cosine_similarities[best_idx] > 0.15:
            relevant_sentences = [sentences[best_idx]]
    
    return relevant_sentences

def load_data():
    if os.path.exists(FILE_PATH):
        df = pd.read_csv(FILE_PATH)
        print(f"Data Loaded: {df.head()}")  # Debugging line
        return df
    else:
        raise FileNotFoundError(f"File not found: {FILE_PATH}")

def process_query(query):
    query = query.lower().strip()
    print(f"Query: {query}")  # Debugging line

    # Predefined responses
    greetings = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! How can I help you?",
        "good morning": "Good morning! What can I do for you today?",
        "good afternoon": "Good afternoon! How can I assist you?",
        "good evening": "Good evening! How can I help you?",
        "how are you": "I'm just a bot, but I'm here to help you!",
        "what is your name": "I'm a chatbot designed to assist you with queries related to the Department of Justice."
    }
    
    if query in greetings:
        return greetings[query]
    
    df = load_data()
    
    if 'text' not in df.columns:
        return "The CSV file must contain a 'text' column."

    sentences = []
    lemmatized_sentences = []

    for doc in df['text']:
        for sentence in split_into_sentences(doc):
            sentences.append(sentence.strip())
            lemmatized_sentences.append(lemmatize_sentence(sentence.strip()))

    print(f"Sentences: {sentences}")  # Debugging line
    print(f"Lemmatized Sentences: {lemmatized_sentences}")  # Debugging line

    corrected_query = correct_spelling(query)
    relevant_sentences = find_relevant_sentences(corrected_query, sentences, lemmatized_sentences)

    if relevant_sentences:
        return " ".join(relevant_sentences)
    else:
        return "No relevant results found."

@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            response = process_query(query)
            chat_history.append({"type": "user-msg", "text": query})
            chat_history.append({"type": "bot-msg", "text": response})

    return render_template("chat.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)
