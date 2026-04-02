import streamlit as st
from backend import (
    extract_video_id,
    get_transcript,
    build_vectorstore,
    answer_question
)

st.set_page_config(page_title="YouTube RAG App", layout="wide")

st.title("🎥 YouTube RAG Q&A App")


youtube_url = st.text_input("Enter YouTube URL")
question = st.text_input("Ask a question")


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ----------------------------
# Process Video
# ----------------------------
if st.button("Process Video"):
    if not youtube_url:
        st.error("Enter a valid URL")
    else:
        video_id = extract_video_id(youtube_url)

        if not video_id:
            st.error("Invalid YouTube URL")
        else:
            st.success(f"Video ID: {video_id}")

            with st.spinner("Fetching transcript..."):
                transcript = get_transcript(video_id)

            with st.spinner("Building vector DB..."):
                st.session_state.vector_store = build_vectorstore(transcript)

            st.success("Ready! Ask questions now 🚀")


# ----------------------------
# Ask Question
# ----------------------------
if st.button("Ask"):
    if st.session_state.vector_store is None:
        st.error("Process a video first")
    elif not question:
        st.error("Enter a question")
    else:
        with st.spinner("Thinking..."):
            answer = answer_question(
                st.session_state.vector_store,
                question
            )

        st.subheader("Answer")
        st.write(answer)