import re


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "what", "how", "why", "when", "where", "which",
    "and", "or", "of", "to", "in", "on", "for",
    "with", "did", "do", "does", "this", "that",
    "these", "those"
}


def tokenize(text):
    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )

    return [
        word
        for word in words
        if word not in STOP_WORDS
    ]


def find_best_evidence(
    claim,
    documents
):
    """
    Find the most relevant document chunk
    for a factual claim.
    """

    if not documents:
        return None

    claim_words = set(
        tokenize(claim)
    )

    if not claim_words:
        return documents[0]

    best_document = None
    best_score = -1

    for document in documents:

        document_words = set(
            tokenize(
                document.page_content
            )
        )

        overlap = (
            claim_words
            & document_words
        )

        score = len(overlap)

        if score > best_score:
            best_score = score
            best_document = document

    return best_document