import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import os
from langchain_groq import ChatGroq

# AI Model Initialization
groq_api_key = st.secrets["general"]["GROQ_API_KEY"]

model = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    groq_api_key=groq_api_key,
)

# Initialize Whisper model
whisper_model = whisper.load_model("small")

def process_audio_file(audio_file):
    # Transcribe the uploaded audio file
    result = whisper_model.transcribe(audio_file, language="en")
    return result['text']

def main():
    st.set_page_config(page_title="Real-Time Speech to Text", layout="centered")

    st.markdown("<h1 style='text-align: center;'>Real-Time Speech to Text 🎙️</h1>", unsafe_allow_html=True)

    # Create tabs for recording and uploading audio
    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])

    # Tab for recording audio
    with tab1:
        st.write("Press the button below to start recording:")
        audio_data = mic_recorder(
            start_prompt="🔴 Start Recording", 
            stop_prompt="⏹️ Stop Recording", 
            key="mic_recorder"
        )

        if audio_data:
            st.success("Recording finished! Processing audio...")

            # Save the recorded audio bytes to a file
            audio_file = "output.wav"
            with open(audio_file, "wb") as f:
                f.write(audio_data['bytes'])

            # Transcribe the recorded audio
            transcription = process_audio_file(audio_file)

            # Check if transcription is not empty
            if transcription.strip():
                st.write("🎧 Transcription complete! Now correcting the text...")
                
                # Correct the transcription using AI model
                complete_query = f"Please correct any typos or grammatical errors in the following text: \"{transcription}\". Provide a coherent and polished version. Just give the corrected text without any additional information."
                response = model.invoke(complete_query)  # Direct invocation
                corrected_text = response.content
            else:
                corrected_text = "No transcription available."

            # Display both raw transcription and corrected text side-by-side
            col1, col2 = st.columns(2)

            # Raw Transcription
            with col1:
                with st.expander("🔍 Raw Transcription", expanded=True):
                    st.markdown(f"<p style='font-size:18px;'>{transcription}</p>", unsafe_allow_html=True)
                    st.code(transcription, language='')  # Text area for copying

            # Corrected Transcription
            with col2:
                with st.expander("📝 Corrected Transcription", expanded=True):
                    st.markdown(f"<p style='font-size:18px;'>{corrected_text}</p>", unsafe_allow_html=True)
                    st.code(corrected_text, language='')  # Text area for copying

            # Automatically delete the audio file after transcription
            os.remove(audio_file)

    # Tab for uploading audio
    with tab2:
        st.write("Upload your audio file (WAV or supported format):")
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg"])

        if uploaded_file is not None:
            st.success("File uploaded successfully! Processing audio...")
            # Save the uploaded file temporarily
            with open("uploaded_audio", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Transcribe the uploaded audio file
            transcription = process_audio_file("uploaded_audio")

            # Check if transcription is not empty
            if transcription.strip():
                st.write("🎧 Transcription complete! Now correcting the text...")
                
                # Correct the transcription using AI model
                complete_query = f"Please correct any typos or grammatical errors in the following text: \"{transcription}\". Provide a coherent and polished version. Just give the corrected text without any additional information."
                response = model.invoke(complete_query)  # Direct invocation
                corrected_text = response.content
            else:
                corrected_text = "No transcription available."

            # Display both raw transcription and corrected text side-by-side
            col1, col2 = st.columns(2)

            # Raw Transcription
            with col1:
                with st.expander("🔍 Raw Transcription", expanded=True):
                    st.markdown(f"<p style='font-size:18px;'>{transcription}</p>", unsafe_allow_html=True)
                    st.code(transcription, language='')  # Text area for copying

            # Corrected Transcription
            with col2:
                with st.expander("📝 Corrected Transcription", expanded=True):
                    st.markdown(f"<p style='font-size:18px;'>{corrected_text}</p>", unsafe_allow_html=True)
                    st.code(corrected_text, language='')  # Text area for copying

            # Automatically delete the uploaded audio file after transcription
            os.remove("uploaded_audio")

    # Add some styling and animations
    st.markdown("""
        <style>
            .stButton > button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #45a049;
            }
        </style>
    """, unsafe_allow_html=True)

    # Instructions Section with Emojis
    st.markdown("""
        <div style="text-align: center;">
            <h2>🤔 How it Works</h2>
            <p>1. Use the "Record Audio" tab to record your voice.</p>
            <p>2. Use the "Upload Audio" tab to upload your own audio file.</p>
            <p>3. The app will process the audio and display both the transcription and corrected text side-by-side.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()