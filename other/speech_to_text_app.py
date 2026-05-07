import streamlit as st
import speech_recognition as sr
import moviepy.editor as mpy
from PIL import ImageTk, Image
from deep_translator import GoogleTranslator

# ------------------- NLTK FIX -------------------
import nltk
# Make sure NLTK knows where your data is
nltk.data.path.append("nltk_data")

# Download required NLTK resources if missing
nltk.download('punkt', download_dir="nltk_data")
nltk.download('punkt_tab', download_dir="nltk_data")
nltk.download('stopwords', download_dir="nltk_data")

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.tokenize import word_tokenize as nltk_word_tokenize

def word_tokenize(text):
    return nltk_word_tokenize(text, preserve_line=True)
# ------------------- END NLTK FIX -------------------


# MoviePy shortcuts
VideoFileClip = mpy.VideoFileClip
concatenate_videoclips = mpy.concatenate_videoclips


# List of keywords
list_of_words = ["acidity","hospital","fasten","bandage","heartbreaking",
                 "digestion","sneeze","nurse","cotton","drink","medicine",
                 "pill","bp","symptoms","doctor","toilet","tall"]

# ------------------- Speech-to-Text Function -------------------
def speech_to_text():
    r = sr.Recognizer()
   
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        st.info("Please speak into the microphone...")
        audio = r.listen(source, timeout=10)

    try:
        text = r.recognize_google(audio, language='te-IN')
        st.success("Speech Recognition Result:")
        st.write(text)

        # Translate and display the translated text
        translated_text = translate_to_english(text)
        st.info("Translated Text:")
        st.write(translated_text)

        # Preprocess the translated text
        clean_text = preprocess_text(translated_text)
        st.write(clean_text)

        # Split into words for video generation
        play_seq = clean_text.split()
        found = 0

        if len(play_seq) == 1:
            # Single word: check if in keyword list
            for i in list_of_words:
                if i == clean_text:
                    found = 1
                    break
            if found == 1:
                # Load the video corresponding to the speech
                video_file = open('dataset/{}.mp4'.format(clean_text), 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
            else:
                st.error(f"Failed to load video for {clean_text}")

        else:
            # Multi-word: concatenate videos
            clips = []
            for i in play_seq:
                cnt = list_of_words.count(i)
                if cnt == 1:
                    video_file = VideoFileClip('dataset/{}.mp4'.format(i))
                    clips.append(video_file)

            # Concatenate clips
            final = concatenate_videoclips(clips, method='compose')

            # Save the final video
            final_path = 'dataset/{}.mp4'.format(clean_text)
            final.write_videofile(final_path, codec='libx264')

            try:
                # Load the final concatenated video
                video_file = open('dataset/{}.mp4'.format(clean_text), 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)

            except:
                st.error("Failed to load avatar, the word is not available in the dataset")

    except sr.UnknownValueError:
        st.warning("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")


# ------------------- Translate Function -------------------
def translate_to_english(text):
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated.lower()
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return text.lower()

# ------------------- Text Preprocessing Function -------------------
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

# ------------------- Main Streamlit App -------------------
def main():
    st.title("SignCare: Virtual Gesture Communication for Hearing-Impaired in Telugu States")
    st.write(
        "This is a  speech-to-avatar application using Streamlit and SpeechRecognition. "
        "Click the button below and start speaking into your microphone."
    )

    if st.button("Start Speech Recognition"):
        speech_to_text()
    

if __name__ == "__main__":
    main()