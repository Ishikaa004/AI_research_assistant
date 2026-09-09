from langchain_community.vectorstores import FAISS


def create_vector_store(
    documents,
    embedding_model
):
    """
    Create a fresh FAISS vector store
    from the supplied documents.
    """

    if not documents:

        raise ValueError(
            "No documents available to create FAISS index."
        )

    vector_store = FAISS.from_documents(
        documents,
        embedding_model
    )

    return vector_store