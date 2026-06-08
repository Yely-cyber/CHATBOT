import string
import random
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Cargar y procesar el corpus
@st.cache_resource
def cargar_corpus():
    archivo = open('Corpus_crucero.txt', 'r', encoding='latin-1', errors='ignore')
    raw = archivo.read().lower()
    sent_tokens = nltk.sent_tokenize(raw)
    return sent_tokens

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

SALUDOS_INPUTS = ("hola", "buenas", "saludos", "qué tal", "hey", "buenos días")
SALUDOS_OUTPUTS = ["Hola", "Hola, ¿Qué tal?", "Hola, ¿Cómo te puedo ayudar?", "Hola, encantado de hablar contigo"]

def saludos(sentence):
    for word in sentence.split():
        if word.lower() in SALUDOS_INPUTS:
            return random.choice(SALUDOS_OUTPUTS)

def response(user_response, sent_tokens):
    tokens_temp = sent_tokens.copy()
    tokens_temp.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stopwords.words('spanish'))
    tfidf = TfidfVec.fit_transform(tokens_temp)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        return "Lo siento, no te entendí. Póngase en contacto con el personal asistencial."
    else:
        return tokens_temp[idx]

# --- Interfaz Streamlit ---
st.title("🚢 Chatbot de Crucero")
st.caption("Hazme preguntas sobre tus vacaciones en el crucero")

sent_tokens = cargar_corpus()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"rol": "bot", "texto": "Hola, contestaré a tus preguntas acerca de tus vacaciones en el crucero. Si deseas salir, escribe *salir*."}]

for msg in st.session_state.mensajes:
    if msg["rol"] == "bot":
        with st.chat_message("assistant"):
            st.write(msg["texto"])
    else:
        with st.chat_message("user"):
            st.write(msg["texto"])

user_input = st.chat_input("Escribe tu pregunta aquí...")

if user_input:
    user_lower = user_input.lower()
    st.session_state.mensajes.append({"rol": "user", "texto": user_input})

    if user_lower in ('salir', 'adios', 'chau'):
        bot_reply = "Nos vemos pronto, ¡Cuídate!"
    elif user_lower in ('gracias', 'muchas gracias'):
        bot_reply = "No hay de qué 😊"
    elif saludos(user_lower):
        bot_reply = saludos(user_lower)
    else:
        bot_reply = response(user_lower, sent_tokens)

    st.session_state.mensajes.append({"rol": "bot", "texto": bot_reply})
    st.rerun()
