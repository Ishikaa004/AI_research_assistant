import os

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
# 1. ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# 2. GLOBAL VARIABLES
# ============================================================

documents = []
chunks = []

bm25 = None
vector_store = None
rag_chain = None
llm = None

memory = ConversationMemory()


# ============================================================
# 3. CREATE LLM + EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = create_embedding_model()

print("Embedding model loaded!")


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

print("LLM created!")


rag_chain = create_rag_chain(llm)

print("RAG chain created!")


# ============================================================
# 4. LOAD DOCUMENTS
# ============================================================

def initialize_pipeline(document_paths):
    """
    Load user-provided documents and create the complete
    retrieval pipeline.

    document_paths:
        List of PDF/TXT file paths.
    """

    global documents
    global chunks
    global bm25
    global vector_store

    documents = []
    chunks = []

    # --------------------------------------------------------
    # LOAD DOCUMENTS
    # --------------------------------------------------------

    for file_path in document_paths:

        file_path = file_path.strip()

        if not file_path:
            continue

        if not os.path.exists(file_path):

            print(
                f"Warning: File not found: {file_path}"
            )

            continue

        filename = os.path.basename(file_path)

        print(
            f"\nLoading: {filename}"
        )

        loaded_documents = load_document(
            file_path
        )

        documents.extend(
            loaded_documents
        )

    # --------------------------------------------------------
    # CHECK DOCUMENTS
    # --------------------------------------------------------

    if not documents:

        raise ValueError(
            "No readable documents were loaded."
        )

    print(
        f"\nOriginal documents: {len(documents)}"
    )

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    chunks = split_documents(
        documents
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25 = create_bm25_index(
        chunks
    )

    print(
        "BM25 index created!"
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    print(
        "FAISS vector store ready!"
    )

    return True


# ============================================================
# 5. CHECK WHETHER PIPELINE IS READY
# ============================================================

def pipeline_ready():

    return (
        len(documents) > 0
        and len(chunks) > 0
        and bm25 is not None
        and vector_store is not None
    )


# ============================================================
# 6. EVIDENCE SUFFICIENCY CHECK
# ============================================================

def check_evidence(
    llm,
    question,
    context
):
    """
    Determine whether the retrieved context contains
    enough information to answer the question.

    This check happens BEFORE answer generation.
    """

    evidence_prompt = f"""
You are an evidence sufficiency checker for a
retrieval-augmented generation system.

Your ONLY task is to determine whether the provided
CONTEXT contains enough information to answer the
QUESTION.

Do NOT answer the question.

Do NOT use your general knowledge.

Do NOT use outside knowledge.

The CONTEXT is the ONLY source of evidence.

RULES:

1. Return SUPPORTED if the context contains enough
   explicit information to answer the question.

2. Return SUPPORTED if the answer can be directly
   synthesized by combining multiple facts from the
   context.

3. Return UNSUPPORTED if the context only mentions
   the topic/entity but does not provide enough
   information to answer the question.

4. Return UNSUPPORTED if answering requires outside
   knowledge.

5. Return UNSUPPORTED if important information required
   by the question is missing.

6. Do not assume facts that are not present.

7. Do not use the QUESTION itself as evidence.

8. Multiple pieces of evidence from different chunks
   may be combined.

9. The evidence must directly support the specific
   information requested by the question.

   Related information is NOT sufficient.

   For example, if the question asks how a model
   was trained, evidence about how the model was
   compiled is not sufficient.

10. Do not mark evidence as SUPPORTED merely because
    it is about the same topic. The evidence must
    contain the requested fact or facts.

11. If the context says that an entity is used for
    something, you may use that documented purpose.

    You may NOT provide a general definition of the
    entity unless the context provides one.

12. Return ONLY one of:

SUPPORTED

UNSUPPORTED


CONTEXT:

{context}


QUESTION:

{question}
"""

    try:

        result = llm.invoke(
            evidence_prompt
        )

        decision = result.content.strip().upper()

        print(
            "Evidence checker:",
            decision
        )

        # IMPORTANT:
        # Check UNSUPPORTED first because
        # "UNSUPPORTED" contains "SUPPORTED".

        if "UNSUPPORTED" in decision:

            return False

        if decision == "SUPPORTED":

            return True

        # Fail closed if unexpected response.

        return False

    except Exception as e:

        print(
            f"Evidence check error: {e}"
        )

        # Fail closed.

        return False


# ============================================================
# 7. CORE RAG FUNCTION
# ============================================================

def run_rag(
    question,
    source_filter="",
    use_memory=True,
    save_memory=True,
    verbose=True
):
    """
    Run the complete RAG pipeline.

    Returns:

        answer:
            Final generated answer.

        retrieved_documents:
            Top reranked retrieval results BEFORE
            contextual compression.

            These documents are used for retrieval
            evaluation such as Hit@K, Recall@K and MRR.

        context:
            Final compressed context used by the
            evidence checker and answer generator.
    """

    # --------------------------------------------------------
    # CHECK DOCUMENTS
    # --------------------------------------------------------

    if not pipeline_ready():

        return (
            "Please load a document first.",
            [],
            ""
        )

    # --------------------------------------------------------
    # CLEAN QUESTION
    # --------------------------------------------------------

    question = question.strip()

    if not question:

        return (
            "I don't know based on the provided documents.",
            [],
            ""
        )

    # --------------------------------------------------------
    # METADATA FILTERING
    # --------------------------------------------------------

    if source_filter:

        filtered_chunks = filter_documents(
            chunks,
            source_filter
        )

        if verbose:

            print(
                f"\nMetadata filtering: "
                f"{len(filtered_chunks)} chunks matched."
            )

        if not filtered_chunks:

            return (
                "I don't know based on the provided documents.",
                [],
                ""
            )

    else:

        filtered_chunks = chunks

        if verbose:

            print(
                "\nMetadata filtering: All documents."
            )

    # --------------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------------

    if use_memory:

        history = memory.get_history()

        history_text = "\n".join(
            f"{message.type}: {message.content}"
            for message in history
        )

    else:

        history_text = ""

    # --------------------------------------------------------
    # QUERY TRANSFORMATION
    # --------------------------------------------------------

    transformed_question = transform_query(
        llm,
        question,
        history_text
    )

    if verbose:

        print(
            "\n=============================="
        )

        print(
            "QUERY TRANSFORMATION"
        )

        print(
            "=============================="
        )

        print(
            "Original:",
            question
        )

        print(
            "Transformed:",
            transformed_question
        )

    # --------------------------------------------------------
    # MULTI QUERY
    # --------------------------------------------------------

    queries = generate_queries(
        llm,
        transformed_question
    )

    if not queries:

        queries = [
            transformed_question
        ]

    if verbose:

        print(
            "\n=============================="
        )

        print(
            "MULTI-QUERY RETRIEVAL"
        )

        print(
            "=============================="
        )

        for i, query in enumerate(
            queries,
            start=1
        ):

            print(
                f"Query {i}: {query}"
            )

    # --------------------------------------------------------
    # FAISS RETRIEVAL
    # --------------------------------------------------------

    faiss_results_all = []

    for query in queries:

        faiss_results = vector_store.similarity_search(
            query,
            k=5
        )

        if source_filter:

            faiss_results = filter_documents(
                faiss_results,
                source_filter
            )

        faiss_results_all.append(
            faiss_results
        )

    if verbose:

        print(
            "\n=============================="
        )

        print(
            "FAISS SEARCH"
        )

        print(
            "=============================="
        )

        print(
            "FAISS retrieved:",
            sum(
                len(result)
                for result in faiss_results_all
            ),
            "chunks."
        )

    # --------------------------------------------------------
    # BM25 RETRIEVAL
    # --------------------------------------------------------

    bm25_results_all = []

    for query in queries:

        # Use filtered chunks when a source filter
        # is active.

        search_chunks = filtered_chunks

        bm25_results = bm25_search(
            bm25,
            search_chunks,
            query,
            k=5
        )

        if source_filter:

            bm25_results = filter_documents(
                bm25_results,
                source_filter
            )

        bm25_results_all.append(
            bm25_results
        )

    if verbose:

        print(
            "BM25 retrieved:",
            sum(
                len(result)
                for result in bm25_results_all
            ),
            "chunks."
        )

    # --------------------------------------------------------
    # RRF
    # --------------------------------------------------------

    if verbose:

        print(
            "\n=============================="
        )

        print(
            "RAG FUSION (RRF)"
        )

        print(
            "=============================="
        )

    result_lists = []

    result_lists.extend(
        faiss_results_all
    )

    result_lists.extend(
        bm25_results_all
    )

    rrf_results = reciprocal_rank_fusion(
        result_lists
    )

    # --------------------------------------------------------
    # EXTRACT DOCUMENTS
    # --------------------------------------------------------

    results = []

    for item in rrf_results:

        if isinstance(item, tuple):

            document = item[0]

        else:

            document = item

        results.append(
            document
        )

    if verbose:

        print(
            f"RRF ranked: "
            f"{len(results)} unique documents."
        )

    # --------------------------------------------------------
    # RERANKING
    # --------------------------------------------------------

    results = rerank_documents(
        transformed_question,
        results
    )

    # --------------------------------------------------------
    # RETRIEVAL EVALUATION SET
    # --------------------------------------------------------

    # Keep top 8 after reranking.

    results = results[:8]

    # IMPORTANT:
    #
    # These are the actual retrieval results that
    # should be evaluated.
    #
    # We make a separate copy so later compression
    # does not modify the evaluation set.

    retrieved_documents = results.copy()

    if verbose:

        print(
            f"After reranking: "
            f"{len(retrieved_documents)} chunks."
        )

    # --------------------------------------------------------
    # NO RETRIEVAL RESULTS
    # --------------------------------------------------------

    if not retrieved_documents:

        answer = (
            "I don't know based on the provided documents."
        )

        if save_memory:

            memory.add_user_message(
                question
            )

            memory.add_ai_message(
                answer
            )

        return (
            answer,
            [],
            ""
        )

    # --------------------------------------------------------
    # DEBUG RETRIEVED CHUNKS
    # --------------------------------------------------------

    if verbose:

        print(
            "\n=============================="
        )

        print(
            "RETRIEVED CONTEXT BEFORE COMPRESSION"
        )

        print(
            "=============================="
        )

        for i, document in enumerate(
            retrieved_documents,
            start=1
        ):

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                None
            )

            print(
                f"\n--- Chunk {i} ---"
            )

            print(
                "Source:",
                os.path.basename(source)
            )

            print(
                "Page:",
                page
            )

            print(
                document.page_content[:1500]
            )

        print(
            "\n=============================="
        )

    # ========================================================
    # CONTEXTUAL COMPRESSION
    # ========================================================

    compressed_results = compress_documents(
        llm,
        question,
        retrieved_documents
    )

    if verbose:

        print(
            f"\nAfter compression: "
            f"{len(compressed_results)} chunks."
        )

    # --------------------------------------------------------
    # DEBUG COMPRESSED CONTEXT
    # --------------------------------------------------------

    if verbose:

        print(
            "\n=============================="
        )

        print(
            "COMPRESSED CONTEXT"
        )

        print(
            "=============================="
        )

        for i, document in enumerate(
            compressed_results,
            start=1
        ):

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                None
            )

            print(
                f"\n--- Compressed Chunk {i} ---"
            )

            print(
                "Source:",
                os.path.basename(source)
            )

            print(
                "Page:",
                page
            )

            print(
                document.page_content
            )

        print(
            "\n=============================="
        )

    # --------------------------------------------------------
    # NO USEFUL COMPRESSED CONTEXT
    # --------------------------------------------------------

    if not compressed_results:

        answer = (
            "I don't know based on the provided documents."
        )

        if verbose:

            print(
                "\n=============================="
            )

            print(
                "ANSWER"
            )

            print(
                "=============================="
            )

            print(answer)

        if save_memory:

            memory.add_user_message(
                question
            )

            memory.add_ai_message(
                answer
            )

        # IMPORTANT:
        #
        # Retrieval evaluation should still receive
        # retrieved_documents even if compression
        # failed.

        return (
            answer,
            retrieved_documents,
            ""
        )

    # --------------------------------------------------------
    # CREATE FINAL CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for document in compressed_results:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            None
        )

        if page:

            source_info = (
                f"Source: {os.path.basename(source)}, "
                f"Page: {page}"
            )

        else:

            source_info = (
                f"Source: {os.path.basename(source)}"
            )

        context_parts.append(
            f"{source_info}\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # DEBUG FINAL CONTEXT
    # --------------------------------------------------------

    if verbose:

        print(
            "\n=============================="
        )

        print(
            "FINAL CONTEXT SENT TO EVIDENCE CHECKER"
        )

        print(
            "=============================="
        )

        print(context)

        print(
            "\n=============================="
        )

    # ========================================================
    # EVIDENCE SUFFICIENCY CHECK
    # ========================================================

    if verbose:

        print(
            "\n=============================="
        )

        print(
            "EVIDENCE SUFFICIENCY CHECK"
        )

        print(
            "=============================="
        )

    is_supported = check_evidence(
        llm,
        question,
        context
    )

    if verbose:

        print(
            "Evidence sufficient:",
            is_supported
        )

    # --------------------------------------------------------
    # REJECT UNSUPPORTED ANSWER
    # --------------------------------------------------------

    if not is_supported:

        answer = (
            "I don't know based on the provided documents."
        )

        if verbose:

            print(
                "\n=============================="
            )

            print(
                "ANSWER"
            )

            print(
                "=============================="
            )

            print(answer)

        if save_memory:

            memory.add_user_message(
                question
            )

            memory.add_ai_message(
                answer
            )

        # IMPORTANT:
        #
        # Return retrieved_documents, not
        # compressed_results.
        #
        # Retrieval evaluation should evaluate
        # retrieval independently from generation.

        return (
            answer,
            retrieved_documents,
            context
        )

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    if verbose:

        print(
            "\n=============================="
        )

        print(
            "GENERATING ANSWER"
        )

        print(
            "=============================="
        )

    answer = rag_chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": history_text
        }
    )

    answer = answer.strip()

    # --------------------------------------------------------
    # DISPLAY FINAL ANSWER
    # --------------------------------------------------------

    if verbose:

        print(
            "\n=============================="
        )

        print(
            "ANSWER"
        )

        print(
            "=============================="
        )

        print(answer)

    # --------------------------------------------------------
    # SAVE MEMORY
    # --------------------------------------------------------

    if save_memory:

        memory.add_user_message(
            question
        )

        memory.add_ai_message(
            answer
        )

    # --------------------------------------------------------
    # FINAL RETURN
    # --------------------------------------------------------

    return (
        answer,
        retrieved_documents,
        context
    )


# ============================================================
# 8. INTERACTIVE CHAT
# ============================================================

def chat():

    print(
        "\n========================================"
    )

    print(
        "AI RESEARCH ASSISTANT"
    )

    print(
        "========================================"
    )

    print(
        "\nEnter document paths."
    )

    print(
        "You can enter multiple paths separated by commas."
    )

    print(
        "Example:"
    )

    print(
        r"C:\Users\Dell\Documents\research.pdf"
    )

    # --------------------------------------------------------
    # DOCUMENT INPUT
    # --------------------------------------------------------

    while True:

        document_input = input(
            "\nDocument path(s): "
        ).strip()

        if not document_input:

            print(
                "Please enter at least one document path."
            )

            continue

        document_paths = [
            path.strip()
            for path in document_input.split(",")
            if path.strip()
        ]

        try:

            initialize_pipeline(
                document_paths
            )

            break

        except ValueError as e:

            print(
                f"\nError loading documents: {e}"
            )

    # --------------------------------------------------------
    # DOCUMENTS READY
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "DOCUMENTS READY"
    )

    print(
        "========================================"
    )

    for document_path in document_paths:

        print(
            f"- {os.path.basename(document_path)}"
        )

    print(
        "\nYou can now ask questions."
    )

    # --------------------------------------------------------
    # CHAT LOOP
    # --------------------------------------------------------

    while True:

        question = input(
            "\nAsk a question "
            "(type 'exit' to quit): "
        ).strip()

        if question.lower() == "exit":

            print(
                "\nExiting assistant..."
            )

            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        source_filter = input(
            "Filter by file "
            "(press Enter for all): "
        ).strip()

        # ----------------------------------------------------
        # RUN RAG
        # ----------------------------------------------------

        answer, results, context = run_rag(
            question=question,
            source_filter=source_filter,
            use_memory=True,
            save_memory=True,
            verbose=True
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        print(
            "\n=============================="
        )

        print(
            "SOURCES"
        )

        print(
            "=============================="
        )

        # The final context contains the exact
        # compressed evidence used for the answer.

        if context:

            seen_sources = set()

            for line in context.splitlines():

                if not line.startswith("Source:"):

                    continue

                source_line = line.replace(
                    "Source:",
                    "",
                    1
                ).strip()

                if ", Page:" in source_line:

                    source_name, page = source_line.split(
                        ", Page:",
                        1
                    )

                    source_name = source_name.strip()
                    page = page.strip()

                    source_key = (
                        source_name,
                        page
                    )

                    if source_key not in seen_sources:

                        seen_sources.add(
                            source_key
                        )

                        print(
                            f"- {source_name} "
                            f"- Page {page}"
                        )

                else:

                    source_name = source_line

                    source_key = (
                        source_name,
                        None
                    )

                    if source_key not in seen_sources:

                        seen_sources.add(
                            source_key
                        )

                        print(
                            f"- {source_name}"
                        )

        else:

            print(
                "No supporting source found."
            )


# ============================================================
# 9. START CHAT
# ============================================================

if __name__ == "__main__":

    chat()