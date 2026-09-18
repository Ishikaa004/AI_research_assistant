import sys
import os

# ============================================================
# PROJECT ROOT FIX
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import streamlit as st

from app.main import (
    initialize_pipeline,
    run_rag,
    pipeline_ready
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 5%,
                rgba(124, 58, 237, 0.16),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(6, 182, 212, 0.10),
                transparent 28%
            ),
            #080b14;

        color: #f8fafc;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0d111d 0%,
                #080b13 100%
            );

        border-right: 1px solid #20283a;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.8rem;
    }

    .sidebar-brand {
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.4px;
    }

    .sidebar-description {
        color: #8b95aa;
        font-size: 0.82rem;
        line-height: 1.5;
        margin-top: 0.4rem;
        margin-bottom: 1.5rem;
    }

    .sidebar-section {
        color: #aeb8ca;
        font-size: 0.72rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 1.3rem;
        margin-bottom: 0.65rem;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        padding: 0.5rem 0 1.3rem 0;
    }

    .hero-badge {
        display: inline-block;

        padding: 0.35rem 0.8rem;

        border-radius: 999px;

        background: rgba(124, 58, 237, 0.13);

        border: 1px solid rgba(139, 92, 246, 0.35);

        color: #c4b5fd;

        font-size: 0.72rem;

        font-weight: 750;

        margin-bottom: 0.85rem;
    }

    .hero-title {
        font-size: 3rem;

        font-weight: 850;

        line-height: 1.05;

        letter-spacing: -1.8px;

        background:
            linear-gradient(
                90deg,
                #ffffff,
                #c4b5fd,
                #67e8f9
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #8f9ab0;

        font-size: 1rem;

        margin-top: 0.7rem;

        max-width: 720px;

        line-height: 1.6;
    }


    /* ========================================================
       STATUS CARD
       ======================================================== */

    .status-card {
        background:
            linear-gradient(
                135deg,
                rgba(15, 23, 42, 0.95),
                rgba(17, 24, 39, 0.90)
            );

        border: 1px solid #263149;

        border-radius: 18px;

        padding: 1rem 1.2rem;

        margin: 0.5rem 0 1.4rem 0;

        box-shadow:
            0 10px 35px rgba(0, 0, 0, 0.25);
    }

    .status-row {
        display: flex;

        align-items: center;

        gap: 0.7rem;
    }

    .status-dot {
        width: 10px;

        height: 10px;

        min-width: 10px;

        border-radius: 50%;

        background: #22c55e;

        box-shadow:
            0 0 12px rgba(34, 197, 94, 0.7);
    }

    .status-dot.waiting {
        background: #f59e0b;

        box-shadow:
            0 0 12px rgba(245, 158, 11, 0.65);
    }

    .status-title {
        font-weight: 750;

        color: #f8fafc;

        font-size: 0.94rem;
    }

    .status-text {
        color: #8994aa;

        font-size: 0.81rem;

        margin-top: 0.35rem;

        margin-left: 1.7rem;

        line-height: 1.5;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                #111827,
                #0e1421
            );

        border: 1px solid #273249;

        border-radius: 15px;

        padding: 1rem;

        text-align: center;

        min-height: 90px;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.18);
    }

    .metric-icon {
        font-size: 1.35rem;

        margin-bottom: 0.25rem;
    }

    .metric-label {
        color: #8a95aa;

        font-size: 0.7rem;

        font-weight: 700;

        letter-spacing: 0.4px;
    }


    /* ========================================================
       WELCOME CARD
       ======================================================== */

    .welcome-card {
        background:
            linear-gradient(
                145deg,
                rgba(20, 25, 43, 0.97),
                rgba(12, 16, 28, 0.98)
            );

        border: 1px solid #29334a;

        border-radius: 24px;

        padding: 2.8rem 2rem;

        margin: 1.5rem 0;

        text-align: center;

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.30);
    }

    .welcome-icon {
        font-size: 3rem;

        margin-bottom: 0.6rem;
    }

    .welcome-title {
        font-size: 1.65rem;

        font-weight: 800;

        color: #f8fafc;
    }

    .welcome-text {
        color: #8994aa;

        max-width: 650px;

        margin: 0.7rem auto 0 auto;

        line-height: 1.6;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 12px;

        border: 1px solid #303b52;

        background: #111827;

        color: #e5e7eb;

        font-weight: 650;

        min-height: 43px;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #8b5cf6;

        color: #ffffff;

        background: #171d2d;

        box-shadow:
            0 0 18px rgba(139, 92, 246, 0.18);
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    section[data-testid="stFileUploader"] {
        background: #111827;

        border: 1px dashed #364158;

        border-radius: 14px;

        padding: 0.4rem;
    }


    /* ========================================================
       DOCUMENT CARD
       ======================================================== */

    .document-card {
        background: #111827;

        border: 1px solid #273249;

        border-radius: 12px;

        padding: 0.65rem 0.8rem;

        margin: 0.4rem 0;

        color: #cbd5e1;

        font-size: 0.8rem;

        overflow-wrap: anywhere;
    }


    /* ========================================================
       SOURCE CARD
       ======================================================== */

    .source-card {
        background:
            linear-gradient(
                135deg,
                #111827,
                #0d1320
            );

        border: 1px solid #28344b;

        border-radius: 13px;

        padding: 0.8rem 1rem;

        margin: 0.5rem 0;
    }

    .source-name {
        color: #e5e7eb;

        font-weight: 650;

        font-size: 0.86rem;
    }

    .source-page {
        color: #7f8ba2;

        font-size: 0.76rem;

        margin-top: 0.25rem;
    }


    /* ========================================================
       CHAT
       ======================================================== */

    div[data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.55);

        border: 1px solid #222c40;

        border-radius: 18px;

        padding: 0.8rem 1rem;

        margin-bottom: 0.8rem;
    }
     /* ========================================================
   CHAT CONVERSATION TEXT - WHITE
   ======================================================== */

div[data-testid="stChatMessage"] {
    color: #ffffff !important;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] div,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] strong,
div[data-testid="stChatMessage"] em {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

    /* ========================================================
       CHAT INPUT
       ======================================================== */

    div[data-testid="stChatInput"] {
        background: #111827;

        border: 1px solid #303b52;

        border-radius: 16px;
    }

    div[data-testid="stChatInput"] textarea {
        color: #c084fc !important;
        caret-color: #22d3ee !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #8f9ab0 !important;
        opacity: 1 !important;
    }

    div[data-testid="stChatInput"] textarea:focus {
        color: #c084fc !important;
        caret-color: #22d3ee !important;
    }


    /* ========================================================
       EXPANDER
       ======================================================== */

    div[data-testid="stExpander"] {
        background: transparent;

        border: 1px solid #273249;

        border-radius: 13px;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }


    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {
        border-color: #202a3d;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;

        color: #59657a;

        font-size: 0.72rem;

        margin-top: 3rem;

        padding-top: 1rem;

        border-top: 1px solid #202a3d;
    }

</style>
""")


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "documents_ready" not in st.session_state:
    st.session_state.documents_ready = pipeline_ready()

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    st.html("""
    <div class="sidebar-brand">
        🤖 AI Research Assistant
    </div>

    <div class="sidebar-description">
        Research smarter with document-grounded
        Retrieval-Augmented Generation.
    </div>
    """)


    # --------------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------------

    st.html("""
    <div class="sidebar-section">
        Knowledge Base
    </div>
    """)


    uploaded_files = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more research documents."
    )


    # --------------------------------------------------------
    # SELECTED FILES
    # --------------------------------------------------------

    if uploaded_files:

        st.html("""
        <div class="sidebar-section">
            Selected Documents
        </div>
        """)

        for file in uploaded_files:

            st.html(
                f"""
                <div class="document-card">
                    📄 {file.name}
                </div>
                """
            )


        # ----------------------------------------------------
        # BUILD KNOWLEDGE BASE
        # ----------------------------------------------------

        if st.button(
            "⚡ Build Knowledge Base",
            use_container_width=True
        ):

            upload_directory = os.path.join(
                PROJECT_ROOT,
                "data",
                "uploads"
            )

            os.makedirs(
                upload_directory,
                exist_ok=True
            )

            document_paths = []

            progress = st.progress(0)


            try:

                for index, uploaded_file in enumerate(
                    uploaded_files
                ):

                    file_path = os.path.join(
                        upload_directory,
                        uploaded_file.name
                    )

                    with open(
                        file_path,
                        "wb"
                    ) as f:

                        f.write(
                            uploaded_file.getbuffer()
                        )

                    document_paths.append(
                        file_path
                    )

                    progress.progress(
                        int(
                            (
                                (index + 1)
                                / len(uploaded_files)
                            ) * 100
                        )
                    )


                # ------------------------------------------------
                # INITIALIZE PIPELINE
                # ------------------------------------------------

                with st.spinner(
                    "🔄 Building knowledge base..."
                ):

                    initialize_pipeline(
                        document_paths
                    )


                st.session_state.documents_ready = True

                st.session_state.uploaded_files = [
                    file.name
                    for file in uploaded_files
                ]

                st.session_state.messages = []

                st.session_state.pending_question = None

                st.success(
                    "Knowledge base ready!"
                )

                st.rerun()


            except Exception as e:

                st.error(
                    f"Failed to build knowledge base: {e}"
                )


    st.divider()


    # --------------------------------------------------------
    # SIDEBAR STATUS
    # --------------------------------------------------------

    if st.session_state.documents_ready:

        st.html("""
        <div class="status-row">

            <div class="status-dot"></div>

            <div class="status-title">
                Knowledge Base Ready
            </div>

        </div>
        """)

    else:

        st.html("""
        <div class="status-row">

            <div class="status-dot waiting"></div>

            <div class="status-title">
                Waiting for Documents
            </div>

        </div>
        """)


    # --------------------------------------------------------
    # INDEXED DOCUMENTS
    # --------------------------------------------------------

    if st.session_state.uploaded_files:

        st.html("""
        <div class="sidebar-section">
            Indexed Documents
        </div>
        """)

        for filename in st.session_state.uploaded_files:

            st.html(
                f"""
                <div class="document-card">
                    📄 {filename}
                </div>
                """
            )


    st.divider()


    # --------------------------------------------------------
    # CLEAR CONVERSATION
    # --------------------------------------------------------

    if st.button(
        "🧹 Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.pending_question = None

        st.rerun()


    # --------------------------------------------------------
    # SIDEBAR FOOTER
    # --------------------------------------------------------

    st.html("""
    <div style="
        color:#59657a;
        font-size:0.7rem;
        text-align:center;
        margin-top:1.5rem;
        line-height:1.6;
    ">
        RAG • Hybrid Retrieval • RRF<br>
        Reranking • Compression<br>
        Evidence Grounding • NLI
    </div>
    """)


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">

    <div class="hero-badge">
        ✦ DOCUMENT-GROUNDED AI
    </div>

    <div class="hero-title">
        AI Research Assistant
    </div>

    <div class="hero-subtitle">
        Your intelligent research companion for
        querying, understanding and exploring
        your documents with grounded RAG.
    </div>

</div>
""")


# ============================================================
# MAIN STATUS
# ============================================================

if st.session_state.documents_ready:

    st.html("""
    <div class="status-card">

        <div class="status-row">

            <div class="status-dot"></div>

            <div class="status-title">
                Knowledge Base Ready
            </div>

        </div>

        <div class="status-text">
            Documents are indexed and ready.
            Answers are generated using retrieved
            evidence from your knowledge base.
        </div>

    </div>
    """)

else:

    st.html("""
    <div class="status-card">

        <div class="status-row">

            <div class="status-dot waiting"></div>

            <div class="status-title">
                Knowledge Base Not Ready
            </div>

        </div>

        <div class="status-text">
            Upload PDF or TXT documents from the
            sidebar to create your knowledge base.
        </div>

    </div>
    """)


# ============================================================
# FEATURE CARDS
# ============================================================

if st.session_state.documents_ready:

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.html("""
        <div class="metric-card">

            <div class="metric-icon">
                📄
            </div>

            <div class="metric-label">
                DOCUMENT SEARCH
            </div>

        </div>
        """)


    with col2:

        st.html("""
        <div class="metric-card">

            <div class="metric-icon">
                🔎
            </div>

            <div class="metric-label">
                HYBRID RETRIEVAL
            </div>

        </div>
        """)


    with col3:

        st.html("""
        <div class="metric-card">

            <div class="metric-icon">
                🧠
            </div>

            <div class="metric-label">
                GROUNDED GENERATION
            </div>

        </div>
        """)


    with col4:

        st.html("""
        <div class="metric-card">

            <div class="metric-icon">
                🛡️
            </div>

            <div class="metric-label">
                FAITHFULNESS CHECK
            </div>

        </div>
        """)


    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# WELCOME SCREEN
# ============================================================

if (
    st.session_state.documents_ready
    and len(st.session_state.messages) == 0
):

    st.html("""
    <div class="welcome-card">

        <div class="welcome-icon">
            🔬
        </div>

        <div class="welcome-title">
            What would you like to research?
        </div>

        <div class="welcome-text">
            Ask a question about your uploaded documents.
            The system retrieves relevant evidence,
            reranks it and generates a grounded answer.
        </div>

    </div>
    """)


    st.html("""
    <div style="
        color:#cbd5e1;
        font-size:1rem;
        font-weight:700;
        margin-bottom:0.7rem;
    ">
        ⚡ Suggested Questions
    </div>
    """)


    col1, col2, col3 = st.columns(3)


    with col1:

        if st.button(
            "📊 Explain the dataset",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "Explain the dataset."
            )

            st.rerun()


    with col2:

        if st.button(
            "🧠 Explain the model",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "Explain the model architecture."
            )

            st.rerun()


    with col3:

        if st.button(
            "⚙️ How was it trained?",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "How was the model trained?"
            )

            st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                f"📚 View Sources ({len(message['sources'])})"
            ):

                for source in message["sources"]:

                    st.html(
                        f"""
                        <div class="source-card">

                            <div class="source-name">
                                📄 {source["name"]}
                            </div>

                            <div class="source-page">
                                Page {source["page"]}
                            </div>

                        </div>
                        """
                    )


# ============================================================
# CHAT INPUT
# ============================================================

chat_question = st.chat_input(
    "Ask anything about your documents..."
)


# ============================================================
# GET QUESTION
# ============================================================

if st.session_state.pending_question:

    question = st.session_state.pending_question

    st.session_state.pending_question = None

elif chat_question:

    question = chat_question

else:

    question = None


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # CHECK KNOWLEDGE BASE
    # --------------------------------------------------------

    if not st.session_state.documents_ready:

        st.warning(
            "Please upload documents and build the knowledge base first."
        )

        st.stop()


    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching evidence..."
        ):

            try:

                answer, results, context = run_rag(
                    question
                )

            except Exception as e:

                answer = (
                    "Something went wrong while processing "
                    "your question."
                )

                results = []

                context = ""

                st.error(
                    f"Error: {e}"
                )


        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.markdown(answer)


        # ----------------------------------------------------
        # BUILD SOURCES
        # ----------------------------------------------------

        sources = []

        seen = set()


        for document in results:

            source_path = document.metadata.get(
                "source",
                "Unknown source"
            )

            page = document.metadata.get(
                "page",
                "N/A"
            )


            filename = os.path.basename(
                source_path
            )


            key = (
                filename,
                page
            )


            if key not in seen:

                seen.add(key)

                sources.append(
                    {
                        "name": filename,
                        "page": page
                    }
                )


        sources = sources[:5]


        # ----------------------------------------------------
        # SOURCE DISPLAY
        # ----------------------------------------------------

        if sources:

            with st.expander(
                f"📚 View Sources ({len(sources)})"
            ):

                for source in sources:

                    st.html(
                        f"""
                        <div class="source-card">

                            <div class="source-name">
                                📄 {source["name"]}
                            </div>

                            <div class="source-page">
                                Page {source["page"]}
                            </div>

                        </div>
                        """
                    )


    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )


    # --------------------------------------------------------
    # REFRESH
    # --------------------------------------------------------

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">
    AI Research Assistant · Hybrid RAG · BM25 + FAISS ·
    RRF · Reranking · Contextual Compression ·
    Evidence Grounding · NLI Faithfulness
</div>
""")