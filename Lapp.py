import streamlit as st
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
import contractions

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load the pre-trained model, vectorizer, and label encoder
model_MNB = joblib.load('multinomial_nb_model.joblib')
vectorizer = joblib.load('vectorizer.pkl')  # Ensure this is a CountVectorizer

# Initialize WordNetLemmatizer
lm = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    # Remove special characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Fix contractions
    text = ' '.join([contractions.fix(word) for word in text.split()])
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    # Lemmatize
    tokens = [lm.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Define the language classes based on your label encoding
language_classes = ['Arabic', 'Danish', 'Dutch', 'English', 'French', 'German', 'Greek', 'Hindi',
 'Italian', 'Kannada', 'Malayalam', 'Portugeese', 'Russian', 'Spanish',
 'Sweedish', 'Tamil', 'Turkish']  # Replace with your actual classes

st.title('Language Detection App')

# Text input
user_input = st.text_area("Enter text to detect its language:")

if st.button("Predict Language"):
    if user_input:
        # Preprocess the input text
        processed_input = preprocess_text(user_input)
        
        # Vectorize the input using the loaded CountVectorizer
        vectorized_input = vectorizer.transform([processed_input])
        
        # Predict the language class index
        prediction_index = model_MNB.predict(vectorized_input)[0]
        
        # Get the predicted language label based on the index
        predicted_language = language_classes[prediction_index]
        
        st.write(f"Predicted Language: **{predicted_language}**")
    else:
        st.write("Please enter some text to predict.")
