import re
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os


# ----------------------------
# Extract Video ID
# ----------------------------
def extract_video_id(url: str):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None


# ----------------------------
# Get Transcript
# ----------------------------
def get_transcript(video_id):
    transcript_list = YouTubeTranscriptApi().fetch(video_id)
    return " ".join([t.text for t in transcript_list])


# ----------------------------
# Build Vector DB
# ----------------------------
def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)


# ----------------------------
# Ask RAG Question
# ----------------------------
def answer_question(vector_store, question):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = PromptTemplate(
        template="""
                You are a helpful assistant.
                Answer ONLY from the context.

                Context:
                {context}

                Question:
                {question}
                """,
        input_variables=["context", "question"]
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(final_prompt)
    return response.content