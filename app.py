import string
import random
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Asistente de Crucero",
    page_icon="🚢",
    layout="centered",
)

# ── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Header */
.header-box {
    background: linear-gradient(135deg, #0a5c46 0%, #1D9E75 100%);
    border-radius: 16px;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
}
.header-title {
    color: #E1F5EE;
    font-size: 22px;
    font-weight: 600;
    margin: 0;
}
.header-sub {
    color: #9FE1CB;
    font-size: 13px;
    margin: 4px 0 0;
}

/* Burbuja bot */
.bubble-bot {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    font-size: 14px;
    color: #1a1a1a;
    max-width: 80%;
    margin-bottom: 4px;
    line-height: 1.6;
}

/* Burbuja usuario */
.bubble-user {
    background: #1D9E75;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    font-size: 14px;
    color: #E1F5EE;
    max-width: 80%;
    margin-left: auto;
    margin-bottom: 4px;
    line-height: 1.6;
}

.row-bot  { display: flex; justify-content: flex-start; margin-bottom: 8px; }
.row-user { display: flex; justify-content: flex-end;   margin-bottom: 8px; }

.avatar-bot {
    width: 32px; height: 32px; border-radius: 50%;
    background: #0F6E56; color: #E1F5EE;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0; margin-right: 8px; margin-top: 2px;
}

/* Botones de sugerencias */
div.stButton > button {
    background: #E1F5EE;
    color: #085041;
    border: 1px solid #5DCAA5;
    border-radius: 20px;
    font-size: 12px;
    padding: 4px 14px;
    font-family: 'Sora', sans-serif;
    transition: background 0.2s;
}
div.stButton > button:hover {
    background: #9FE1CB;
    color: #04342C;
    border-color: #1D9E75;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f7fdf9;
}
.sidebar-title {
    font-size: 12px;
    color: #5F5E5A;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
    font-weight: 600;
}

/* Input */
[data-testid="stChatInput"] textarea {
    font-family: 'Sora', sans-serif;
    border-radius: 24px;
}

/* Ocultar footer de Streamlit */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Carga del corpus ─────────────────────────────────────────────────────────
@st.cache_resource
def cargar_corpus():
    with open('Corpus_crucero.txt', 'r', encoding='latin-1', errors='ignore') as f:
        raw = f.read().lower()
    return nltk.sent_tokenize(raw)


# ── NLP ──────────────────────────────────────────────────────────────────────
lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = dict((ord(p), None) for p in string.punctuation)

def lem_tokens(tokens):
    return [lemmer.lemmatize(t) for t in tokens]

def lem_normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

SALUDOS_IN  = ("hola", "buenas", "saludos", "qué tal", "hey", "buenos días")
SALUDOS_OUT = [
    "¡Hola! ¿En qué puedo ayudarte?",
    "¡Hola! ¿Cómo te puedo ayudar hoy?",
    "¡Buenas! Estoy aquí para responder tus preguntas sobre el crucero.",
    "¡Hola! Encantado de atenderte. ¿Qué necesitas saber?",
]

def saludo(sentence):
    for word in sentence.split():
        if word.lower() in SALUDOS_IN:
            return random.choice(SALUDOS_OUT)

def respuesta_corpus(user_response, sent_tokens):
    tokens_temp = sent_tokens.copy()
    tokens_temp.append(user_response)
    vec = TfidfVectorizer(tokenizer=lem_normalize, stop_words=stopwords.words('spanish'))
    tfidf = vec.fit_transform(tokens_temp)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx  = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    if flat[-2] == 0:
        return "Lo siento, no encontré información sobre eso. Por favor, contacta al personal del crucero."
    return tokens_temp[idx]

def obtener_respuesta(user_input, sent_tokens):
    texto = user_input.lower().strip()
    if texto in ('salir', 'adios', 'chau'):
        return "¡Hasta pronto! Que disfrutes el viaje. 🚢"
    if texto in ('gracias', 'muchas gracias'):
        return "¡Con mucho gusto! ¿Hay algo más en lo que pueda ayudarte?"
    s = saludo(texto)
    if s:
        return s
    return respuesta_corpus(texto, sent_tokens)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚢 Asistente de Crucero")
    st.markdown("---")
    st.markdown('<div class="sidebar-title">Temas frecuentes</div>', unsafe_allow_html=True)

    temas = {
        "🗺️ Destinos":        "¿Cuáles son los destinos del crucero?",
        "🍽️ Restaurantes":    "¿Cuáles son los horarios del restaurante?",
        "🏊 Actividades":     "¿Qué actividades hay disponibles?",
        "🛏️ Cabinas":         "¿Qué tipos de cabina hay?",
        "📋 Políticas":       "¿Cuáles son las políticas del crucero?",
        "🎭 Entretenimiento": "¿Qué espectáculos hay en el crucero?",
        "🛳️ Embarcación":     "¿A qué hora es el embarque?",
    }

    for label, pregunta in temas.items():
        if st.button(label, key=f"sidebar_{label}", use_container_width=True):
            st.session_state.pregunta_rapida = pregunta

    st.markdown("---")
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.mensajes = []
        st.session_state.pregunta_rapida = ""
        st.rerun()

    st.markdown("---")
    st.caption("Powered by TF-IDF · NLTK")


# ── Header principal ─────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <span style="font-size:36px;">🚢</span>
    <div>
        <p class="header-title">Asistente de Crucero</p>
        <p class="header-sub">Resuelve todas tus dudas sobre tu viaje</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Estado inicial ───────────────────────────────────────────────────────────
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [
        {"rol": "bot", "texto": "¡Hola! Soy tu asistente virtual de crucero. "
                                "Puedes preguntarme sobre destinos, restaurantes, "
                                "actividades, cabinas y más. ¿En qué te puedo ayudar?"}
    ]
if "pregunta_rapida" not in st.session_state:
    st.session_state.pregunta_rapida = ""


# ── Render del historial ─────────────────────────────────────────────────────
sent_tokens = cargar_corpus()

chat_html = ""
for msg in st.session_state.mensajes:
    if msg["rol"] == "bot":
        chat_html += f"""
        <div class="row-bot">
            <div class="avatar-bot">🤖</div>
            <div class="bubble-bot">{msg["texto"]}</div>
        </div>"""
    else:
        chat_html += f"""
        <div class="row-user">
            <div class="bubble-user">{msg["texto"]}</div>
        </div>"""

st.markdown(chat_html, unsafe_allow_html=True)


# ── Sugerencias rápidas ──────────────────────────────────────────────────────
if len(st.session_state.mensajes) <= 1:
    st.markdown("**Preguntas sugeridas:**")
    sugerencias = [
        "¿Cuáles son los destinos?",
        "¿Hay piscina a bordo?",
        "¿Qué incluye el precio?",
        "¿Cuáles son las excursiones?",
    ]
    cols = st.columns(len(sugerencias))
    for col, sug in zip(cols, sugerencias):
        with col:
            if st.button(sug, key=f"sug_{sug}"):
                st.session_state.pregunta_rapida = sug


# ── Input del usuario ────────────────────────────────────────────────────────
user_input = st.chat_input("Escribe tu pregunta aquí...")

# Procesar pregunta rápida (sidebar o sugerencias)
if st.session_state.pregunta_rapida:
    user_input = st.session_state.pregunta_rapida
    st.session_state.pregunta_rapida = ""

if user_input:
    st.session_state.mensajes.append({"rol": "user", "texto": user_input})
    bot_reply = obtener_respuesta(user_input, sent_tokens)
    st.session_state.mensajes.append({"rol": "bot", "texto": bot_reply})
    st.rerun()
