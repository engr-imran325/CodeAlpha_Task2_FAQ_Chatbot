import streamlit as st
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Text preprocessing using NLTK
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = nltk.tokenize.wordpunct_tokenize(text)
    return " ".join(tokens)


# Artificial Intelligence FAQ Dataset
faqs = [
    {
        "question": "What is Artificial Intelligence?",
        "answer": "Artificial Intelligence (AI) is a branch of computer science that enables machines to perform tasks that normally require human intelligence."
    },
    {
        "question": "What are the types of AI?",
        "answer": "The main types of AI are Narrow AI, General AI, and Super AI. Narrow AI performs specific tasks, while General AI and Super AI are theoretical advanced forms."
    },
    {
        "question": "What is Machine Learning?",
        "answer": "Machine Learning is a branch of AI that allows computers to learn patterns from data and make predictions without being explicitly programmed for every task."
    },
    {
        "question": "What is Deep Learning?",
        "answer": "Deep Learning is a subset of Machine Learning that uses multi-layered neural networks to learn complex patterns from large amounts of data."
    },
    {
        "question": "What is a Neural Network?",
        "answer": "A Neural Network is a computing model inspired by the human brain. It consists of interconnected nodes that process information."
    },
    {
        "question": "What is Natural Language Processing?",
        "answer": "Natural Language Processing (NLP) is a field of AI that helps computers understand, interpret, and generate human language."
    },
    {
        "question": "What is Computer Vision?",
        "answer": "Computer Vision is a field of AI that enables computers to understand and analyze images and videos."
    },
    {
        "question": "What is Generative AI?",
        "answer": "Generative AI is a type of AI that creates new content such as text, images, audio, and code."
    },
    {
        "question": "What is ChatGPT?",
        "answer": "ChatGPT is an AI chatbot developed by OpenAI that can understand and generate human-like text."
    },
    {
        "question": "What are the applications of AI?",
        "answer": "AI is used in healthcare, education, banking, transportation, robotics, customer service, and many other industries."
    },
    {
        "question": "What are the advantages of AI?",
        "answer": "AI can automate repetitive tasks, improve accuracy, analyze large amounts of data, and support faster decision-making."
    },
    {
        "question": "What are the disadvantages of AI?",
        "answer": "Disadvantages of AI include high development costs, privacy concerns, possible job displacement, bias, and overdependence on technology."
    },
    {
        "question": "What is Robotics?",
        "answer": "Robotics is a field involving the design, construction, and operation of robots. AI can help robots make intelligent decisions."
    },
    {
        "question": "What is Supervised Learning?",
        "answer": "Supervised Learning is a Machine Learning method in which a model learns from labeled training data."
    },
    {
        "question": "What is Unsupervised Learning?",
        "answer": "Unsupervised Learning is a Machine Learning method that finds patterns and groups in data without labeled answers."
    },
    {
        "question": "What is Reinforcement Learning?",
        "answer": "Reinforcement Learning is a learning method in which an agent learns by taking actions and receiving rewards or penalties."
    },
    {
        "question": "What is an AI model?",
        "answer": "An AI model is a computer program trained on data to recognize patterns, make predictions, or generate content."
    },
    {
        "question": "What is the future of AI?",
        "answer": "AI is expected to improve healthcare, education, automation, scientific research, and many other fields. Responsible development will remain important."
    },
    {
        "question": "Can AI replace humans?",
        "answer": "AI can automate many tasks, but it does not fully replace human creativity, judgment, emotions, and responsibility."
    },
    {
        "question": "What is an AI chatbot?",
        "answer": "An AI chatbot is a software application that communicates with users and provides responses using AI techniques."
    }
]


# Prepare FAQ questions
questions = [faq["question"] for faq in faqs]
processed_questions = [preprocess_text(q) for q in questions]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)


# Find the most similar FAQ
def get_answer(user_question):
    processed_input = preprocess_text(user_question)

    if not processed_input.strip():
        return "Please enter a question.", 0

    user_vector = vectorizer.transform([processed_input])
    similarities = cosine_similarity(user_vector, faq_vectors)

    best_index = similarities.argmax()
    confidence = similarities[0][best_index]

    if confidence < 0.15:
        return (
            "Sorry, I could not find a suitable answer. "
            "Please ask a question related to Artificial Intelligence.",
            confidence
        )

    return faqs[best_index]["answer"], confidence


# Streamlit user interface
st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Artificial Intelligence FAQ Chatbot")
st.write("Ask questions about Artificial Intelligence, Machine Learning, Deep Learning, NLP and more.")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User question
user_input = st.chat_input("Ask your AI-related question...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    answer, confidence = get_answer(user_input)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.write(answer)
        if confidence >= 0.15:
            st.caption(f"Similarity Score: {confidence:.2f}")

# Clear conversation
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun() 