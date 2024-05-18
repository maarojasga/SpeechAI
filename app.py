import streamlit as st
import replicate
import os
from transformers import AutoTokenizer, pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from unidecode import unidecode
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk.data

# Download NLTK resources if needed
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Set the icons for chat messages
icons = {"assistant": "./img/logo_chat.png", "user": "Admin"}

# Application title
st.set_page_config(page_title="SpeechAI")

# Replicate credentials
with st.sidebar:
    st.title('SpeechAI')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter the Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
            st.warning('Please enter your Replicate API token.', icon='⚠️')
            st.markdown(
                "**Don't have an API token** Register at [Replicate]")

    os.environ['REPLICATE_API_TOKEN'] = replicate_api
    st.subheader("Adjusts the model parameters")
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.2, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.85, step=0.01)

# Store LLM-generated abstracts
if "summaries" not in st.session_state.keys():
    st.session_state.summaries = []

# Show or clear chat messages
for summary in st.session_state.summaries:
    with st.chat_message("assistant", avatar=icons["assistant"]):
        st.write(summary)

def clear_summaries():
    st.session_state.summaries = []

st.sidebar.button('Clear summaries', on_click=clear_summaries)
st.sidebar.caption('SpeechAI')

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to ensure that we are not sending too much text to the model"""
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    """Get the number of tokens in a sentence"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

# Function to clean and normalize text
def clean_and_normalize(text):
    """Clean and normalize text to avoid special characters and stopwords."""
    # Convert to lowercase, remove accents and special characters
    text = unidecode(text.lower())
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)

    # Eliminate empty words (stopwords)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Lemmatize each word
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

# Function to generate a summary with adjusted parameters
def generate_summary(text):
    prompt_str = f"summarize: {text}"

    if get_num_tokens(prompt_str) >= 3072:
        st.error("The input text is too long. Please keep it under 3072 tokens.")
        st.button('Clear summaries', on_click=clear_summaries, key="clear_summaries")
        st.stop()

    response = replicate.run(
        "snowflake/snowflake-arctic-instruct",
        input={
            "prompt": prompt_str,
            "prompt_template": r"{prompt}",
            "temperature": temperature,
            "top_p": top_p,
            "max_length": 150    # Compactness level
        })

    if isinstance(response, list):
        response = ' '.join(response)

    return response.strip()

# Function for displaying word frequencies
def plot_word_cloud(text):
    text = clean_and_normalize(text)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Function for sorting according to the labels returned by the model
def classify_sentiment(label):
    if "1 star" in label or "2 stars" in label:
        return "negative"
    elif "4 stars" in label or "5 stars" in label:
        return "positive"
    else:
        return "neutral"

# Main function of sentiment analysis
def analyze_and_display_sentiments(text):
    # Split text into sentences using the NLTK tokenizer
    sentences = nltk.tokenize.sent_tokenize(text, language="english")

    # Load the sentiment analysis pipeline
    classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    # Initialize counters
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    st.subheader("Sentiment Analysis:")
    for sentence in sentences:
        result = classifier(sentence)[0]
        classification = classify_sentiment(result['label'])
        st.write(f"Text: {sentence}\nLabel: {result['label']}, Score: {result['score']:.2f}, Clasification: {classification}")

        # Counting each type of feeling
        if classification == "positive":
            positive_count += 1
        elif classification == "negative":
            negative_count += 1
        else:
            neutral_count += 1

    # Show final percentages
    total_sentences = len(sentences)
    positive_percentage = (positive_count / total_sentences) * 100
    negative_percentage = (negative_count / total_sentences) * 100
    neutral_percentage = (neutral_count / total_sentences) * 100

    st.write(f"\nPositive: {positive_percentage:.2f}%")
    st.write(f"Negative: {negative_percentage:.2f}%")
    st.write(f"Neutral: {neutral_percentage:.2f}%")

# Input text provided by the user
prompt = st.text_area("Introduces text to summarize and analyze feelings", disabled=not replicate_api)

if prompt:
    cleaned_prompt = clean_and_normalize(prompt)
    summary = generate_summary(cleaned_prompt)
    st.session_state.summaries.append(summary)
    with st.chat_message("assistant", avatar=icons["assistant"]):
        st.write(summary)

    # Draw the word cloud of the text provided.
    st.write("This is what your words look like:")
    plot_word_cloud(cleaned_prompt)

    # Analyzing and showing feelings
    analyze_and_display_sentiments(prompt)
