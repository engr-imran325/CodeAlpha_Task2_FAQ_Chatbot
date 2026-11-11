Artificial Intelligence FAQ Chatbot

Project Overview
The Artificial Intelligence FAQ Chatbot is a Python-based application that answers frequently asked questions related to Artificial Intelligence. It uses Natural Language Processing (NLP) and text similarity techniques to identify the most relevant FAQ and provide an appropriate response.

Features
- Answers frequently asked questions about Artificial Intelligence.
- Text preprocessing using NLTK.
- TF-IDF vectorization for text representation.
- Cosine Similarity to find the best matching FAQ.
- Interactive chatbot interface using Streamlit.
- Displays a similarity score for matched questions.
- Chat history and Clear Chat functionality.
- Handles questions that are not related to the FAQ dataset.

Technologies Used
- Python
- Streamlit
- NLTK
- Scikit-learn
- TF-IDF
- Cosine Similarity
  
Installation

1. Clone or download this project.
2. Open the project folder in VS Code.
3. Install the required dependencies:

pip install -r requirements.txt

4. Run the application:

streamlit run app.py

5. Open the local URL displayed in the terminal.

How It Works

1. The user enters a question in the chatbot.
2. NLTK preprocesses the input text.
3. TF-IDF converts the question into numerical vectors.
4. Cosine Similarity compares the question with stored FAQs.
5. The chatbot displays the answer with the highest similarity score.
6. If no suitable match is found, the chatbot displays a fallback response.

FAQ Topics

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Neural Networks
- Natural Language Processing
- Computer Vision
- Generative AI
- ChatGPT
- AI Applications
- Supervised and Unsupervised Learning
- Reinforcement Learning

Project Structure

CodeAlpha_Task2_FAQ_Chatbot/
│
├── app.py
├── requirements.txt
└── README.md

Internship Task

CodeAlpha Artificial Intelligence Internship

Author
Muhammad Imran Shah
