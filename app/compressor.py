from langchain_core.documents import Document


def compress_documents(
    llm,
    question,
    documents
):
    """
    Extractive contextual compression.

    Keeps only information from each document that is
    directly relevant to the question.

    The model must NOT summarize, paraphrase, combine,
    or introduce new information.
    """

    print(f"Compression input: {len(documents)} chunks")

    if not documents:
        return []

    compressed_documents = []

    for doc in documents:

        prompt = f"""
You are an EXTRACTIVE contextual compression component
in a RAG system.

Your task is to extract the exact sentences or bullet
points from the document that are relevant to the question.

IMPORTANT:

You are NOT a summarizer.

You are NOT allowed to rewrite the information.

You are NOT allowed to paraphrase.

You are NOT allowed to combine multiple facts.

You are NOT allowed to infer anything.

You are NOT allowed to add information from your
general knowledge.

You may ONLY copy text that already exists in the
document.

RULES:

1. Copy only sentences, bullet points, table rows,
   or phrases that directly help answer the question.

2. Preserve the original wording as much as possible.

3. Do not create new statements.

4. Do not create conclusions such as:
   "These are the only services mentioned."

5. If several parts of the document are relevant,
   include all relevant parts.

6. If nothing is relevant, return exactly:

NO_RELEVANT_INFORMATION

Question:
{question}

Document:
{doc.page_content}
"""

        try:

            response = llm.invoke(prompt)

            compressed_text = response.content.strip()

            if compressed_text == "NO_RELEVANT_INFORMATION":
                continue

            if not compressed_text:
                continue

            compressed_documents.append(
                Document(
                    page_content=compressed_text,
                    metadata=doc.metadata
                )
            )

        except Exception as e:

            print(
                f"Compression error: {e}"
            )

    print(
        f"Compression output: "
        f"{len(compressed_documents)} chunks"
    )

    return compressed_documents