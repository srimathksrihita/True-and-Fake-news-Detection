import streamlit as st
import re
import joblib
import nltk
from nltk.corpus import stopwords
import warnings
from nltk.stem import PorterStemmer

nltk.download('stopwords')
warnings.filterwarnings('ignore')

tfidf_loaded = joblib.load("tf-idf.joblib")
model_loaded = joblib.load("model.joblib")

ps = PorterStemmer()


def preprocessing(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9 ]', '', text)
    text = text.split()
    text = [word for word in text if word not in set(stopwords.words('english'))]
    text = [ps.stem(word) for word in text]
    preprocessed_text = ' '.join(text)
    return preprocessed_text


def main():
    st.title("**Fake News Detection**")
    st.write("Enter a news article to determine if it's real or fake news.")
    text = st.text_area("Enter the news text:")

    # Preprocess the input text
    vectorized_text = tfidf_loaded.transform([text])

    # Classify the text
    prediction = (model_loaded.predict(vectorized_text)[0])
    print(prediction)
    if st.button("Predict"):
        if text.strip() == "":  # Checking if the text input is empty
            st.warning("Please enter a news article.")
        else:
            if prediction == 1:
                st.success("The news is likely real.")
            else:
                st.error("The News article may be Fake.")


if __name__ == "__main__":
    main()
