import os

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.compressor import compress_documents
from app.loader import load_document
from app.chunker import split_documents
from app.embedder import create_embedding_model
from app.vectorstore import create_vector_store
from app.rag import create_rag_chain
from app.reranker import rerank_documents
from app.query_transform import transform_query
from app.multi_query import generate_queries
from app.bm25 import create_bm25_index, bm25_search
from app.rrf import reciprocal_rank_fusion
from app.metadata_filter import filter_documents
from app.memory import ConversationMemory


# ============================================================
# ENVIRONMENT / API KEY
# ============================================================

load_dotenv()

# First try Streamlit Cloud Secrets
groq_api_key = None

try:
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception as e:
    print(f"Streamlit Secrets error: {e}")

# If Streamlit Secret is not available, try local .env
if not groq_api_key:
    groq_api_key = os.getenv("GROQ_API_KEY")


# ============================================================
# SAFE API KEY CHECK
# ============================================================

if not groq_api_key:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Add it to Streamlit Secrets or your local .env file."
    )

    # Safe diagnostic
    try:
        available_secrets = list(st.secrets.keys())

        if available_secrets:
            st.warning(
                f"Streamlit Secrets detected: {available_secrets}"
            )
        else:
            st.warning(
                "No Streamlit Secrets were detected."
            )
    except Exception:
        st.warning(
            "Streamlit Secrets could not be accessed."
        )

    st.stop()


# ============================================================
# CREATE LLM
# ============================================================

try:
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=groq_api_key
    )

    print("LLM created successfully!")

except Exception as e:
    st.error("Failed to initialize Groq LLM.")
    st.exception(e)
    st.stop()


# ============================================================
# GLOBAL VARIABLES
# ============================================================

documents = []
chunks = []

bm25 = None
vector_store = None
rag_chain = None

memory = ConversationMemory()


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = create_embedding_model()


# ============================================================
# CREATE RAG CHAIN
# ============================================================

rag_chain = create_rag_chain(llm)


# ============================================================
# PIPELINE INITIALIZATION
# ============================================================

def initialize_pipeline(file_paths):

    global documents
    global chunks
    global bm25
    global vector_store

    print("\n==============================")
    print("Initializing RAG Pipeline")
    print("==============================")

    documents = []

    # --------------------------------------------------------
    # LOAD DOCUMENTS
    # --------------------------------------------------------

    for file_path in file_paths:

        print(f"\nLoading: {file_path}")

        loaded_docs = load_document(file_path)

        documents.extend(loaded_docs)

    print(
        f"Loaded documents: {len(documents)}"
    )

    # --------------------------------------------------------
    # CHUNK DOCUMENTS
    # --------------------------------------------------------

    chunks = split_documents(documents)

    print(
        f"Created chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # CREATE BM25 INDEX
    # --------------------------------------------------------

    bm25 = create_bm25_index(chunks)

    print("BM25 index created.")

    # --------------------------------------------------------
    # CREATE FAISS VECTOR STORE
    # --------------------------------------------------------

    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    print("FAISS vector store created.")

    print("\n==============================")
    print("Pipeline Ready")
    print("==============================\n")


# ============================================================
# CHECK PIPELINE STATUS
# ============================================================

def pipeline_ready():

    return (
        len(chunks) > 0
        and bm25 is not None
        and vector_store is not None
        and rag_chain is not None
    )


# ============================================================
# EVIDENCE CHECK
# ============================================================

def check_evidence(question, documents):

    if not documents:
        return False

    question_words = set(
        question.lower().split()
    )

    for document in documents:

        document_words = set(
            document.page_content.lower().split()
        )

        overlap = question_words & document_words

        if len(overlap) >= 1:
            return True

    return False


# ============================================================
# RUN RAG
# ============================================================

def run_rag(question):

    global documents
    global chunks
    global bm25
    global vector_store
    global rag_chain

    if not pipeline_ready():

        return (
            "Please upload and index a document first.",
            [],
            ""
        )

    print("\n")
    print("===================================")
    print("QUESTION")
    print(question)
    print("===================================")

    # ========================================================
    # METADATA FILTER
    # ========================================================

    filtered_chunks = filter_documents(
        chunks,
        question
    )

    if not filtered_chunks:
        filtered_chunks = chunks

    print(
        f"Metadata filtered chunks: "
        f"{len(filtered_chunks)}"
    )

    # ========================================================
    # CONVERSATION HISTORY
    # ========================================================

    chat_history = memory.get_history()

    # ========================================================
    # QUERY TRANSFORMATION
    # ========================================================

    transformed_query = transform_query(
        llm,
        question,
        chat_history
    )

    print(
        f"\nTransformed query:\n"
        f"{transformed_query}"
    )

    # ========================================================
    # MULTI QUERY
    # ========================================================

    queries = generate_queries(
        llm,
        transformed_query
    )

    if not queries:
        queries = [transformed_query]

    print("\nGenerated queries:")

    for query in queries:
        print(f"- {query}")

    # ========================================================
    # HYBRID RETRIEVAL
    # ========================================================

    vector_results = []
    bm25_results = []

    for query in queries:

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        try:

            faiss_docs = vector_store.similarity_search(
                query,
                k=5
            )

            vector_results.extend(
                faiss_docs
            )

        except Exception as e:

            print(
                f"FAISS retrieval error: {e}"
            )

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        try:

            bm25_docs = bm25_search(
                bm25,
                query,
                k=5
            )

            bm25_results.extend(
                bm25_docs
            )

        except Exception as e:

            print(
                f"BM25 retrieval error: {e}"
            )

    print(
        f"\nFAISS results: "
        f"{len(vector_results)}"
    )

    print(
        f"BM25 results: "
        f"{len(bm25_results)}"
    )

    # ========================================================
    # RRF FUSION
    # ========================================================

    results = reciprocal_rank_fusion(
        vector_results,
        bm25_results
    )

    print(
        f"RRF results: "
        f"{len(results)}"
    )

    # ========================================================
    # RERANKING
    # ========================================================

    results = rerank_documents(
        transformed_query,
        results
    )

    print(
        f"Reranked results: "
        f"{len(results)}"
    )

    # Keep top results
    results = results[:8]

    # ========================================================
    # SAVE RETRIEVED DOCUMENTS
    # BEFORE COMPRESSION
    # ========================================================

    retrieved_documents = results.copy()

    # ========================================================
    # CONTEXTUAL COMPRESSION
    # ========================================================

    compressed_results = compress_documents(
        llm,
        transformed_query,
        results
    )

    print(
        f"Compressed results: "
        f"{len(compressed_results)}"
    )

    # ========================================================
    # BUILD FINAL CONTEXT
    # ========================================================

    context_parts = []

    for document in compressed_results:

        context_parts.append(
            document.page_content
        )

    context = "\n\n".join(
        context_parts
    )

    # ========================================================
    # EVIDENCE CHECK
    # ========================================================

    evidence_available = check_evidence(
        transformed_query,
        retrieved_documents
    )

    if not evidence_available:

        answer = (
            "I don't know based on the provided documents."
        )

        memory.add_message(
            question,
            answer
        )

        return (
            answer,
            retrieved_documents,
            context
        )

    # ========================================================
    # RAG GENERATION
    # ========================================================

    try:

        answer = rag_chain.invoke(
            {
                "question": question,
                "context": context,
                "chat_history": chat_history
            }
        )

    except Exception as e:

        print(
            f"RAG generation error: {e}"
        )

        answer = (
            "Something went wrong while generating "
            "the answer."
        )

    # ========================================================
    # SAVE MEMORY
    # ========================================================

    memory.add_message(
        question,
        answer
    )

    # ========================================================
    # RETURN
    # ========================================================

    return (
        answer,
        retrieved_documents,
        context
    )


# ============================================================
# CHAT FUNCTION
# ============================================================

def chat():

    print(
        "\nAI Research Assistant started."
    )

    while True:

        question = input(
            "\nAsk a question: "
        )

        if question.lower() in [
            "exit",
            "quit",
            "bye"
        ]:

            print(
                "Goodbye!"
            )

            break

        if not question.strip():

            continue

        try:

            answer, results, context = run_rag(
                question
            )

            print(
                "\nANSWER:"
            )

            print(answer)

            print(
                "\nSOURCES:"
            )

            for document in results:

                print(
                    document.metadata
                )

        except Exception as e:

            print(
                f"\nError: {e}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    chat()