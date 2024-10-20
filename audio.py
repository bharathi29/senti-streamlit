import streamlit as st
import speech_recognition as sr
import ffmpeg
import imageio_ffmpeg as iio_ffmpeg
import os
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer and speech recognizer
analyzer = SentimentIntensityAnalyzer()
recognizer = sr.Recognizer()

# Get the FFmpeg executable path from imageio-ffmpeg
ffmpeg_path = iio_ffmpeg.get_ffmpeg_exe()

# Function to convert MP3 to WAV using ffmpeg-python with imageio-ffmpeg's FFmpeg binary
def convert_mp3_to_wav(mp3_file):
    try:
        wav_file = mp3_file.replace(".mp3", ".wav")
        (
            ffmpeg
            .input(mp3_file)
            .output(wav_file)
            .run(cmd=ffmpeg_path, quiet=True, overwrite_output=True)  # Specify path to FFmpeg binary
        )
        return wav_file
    except Exception as e:
        print(f"Error converting MP3 to WAV: {e}")
        return None

# Function to convert audio to text using speech recognition
def convert_audio_to_text(audio_file):
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio."
    except sr.RequestError as e:
        return f"Error with the API: {e}"

# Clean text before sentiment analysis
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower()
    text = text.strip()
    return text

# Calculate sentiment score using VADER
def calculate_sentiment_score(review):
    sentiment = analyzer.polarity_scores(review)
    return sentiment['compound']

# Convert score to sentiment label
def score_to_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Streamlit app
st.title("Audio Sentiment Analyzer")

# File uploader for mp3
uploaded_file = st.file_uploader("Upload an MP3 file", type="mp3")

if uploaded_file is not None:
    # Save uploaded MP3 file temporarily
    mp3_file = uploaded_file.name
    with open(mp3_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert MP3 to WAV using ffmpeg-python
    wav_file = convert_mp3_to_wav(mp3_file)
    
    if wav_file:
        # Convert the WAV file to text using speech recognition
        text = convert_audio_to_text(wav_file)
        st.write(f"Transcribed Text: {text}")

        # Prepare the text data for sentiment analysis
        manual_data = {"text": text}
        df_manual = pd.DataFrame([manual_data])
        df_manual['text'] = df_manual['text'].astype(str).fillna('')
        df_manual['cleaned_review'] = df_manual['text'].apply(clean_text)

        # Calculate sentiment
        df_manual['sentiment_score'] = df_manual['cleaned_review'].apply(calculate_sentiment_score)
        df_manual['label'] = df_manual['sentiment_score'].apply(score_to_label)

        # Display the result
        first_review = df_manual['cleaned_review'].iloc[0]
        first_score = df_manual['sentiment_score'].iloc[0]
        first_label = df_manual['label'].iloc[0]

        st.write(f"Review: {first_review}")
        st.write(f"Sentiment Label: {first_label}")

        # Clean up temporary files
        os.remove(mp3_file)
        os.remove(wav_file)
    else:
        st.error("Failed to convert the MP3 file to WAV.")
