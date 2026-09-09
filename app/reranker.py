import re


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "what", "how", "why", "when", "where", "which",
    "and", "or", "of", "to", "in", "on", "for",
    "with", "did", "do", "does"
}


def tokenize(text):
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    return [
        word
        for word in words
        if word not in STOP_WORDS
    ]


def rerank_documents(question, documents):

    question_words = tokenize(question)

    scored_documents = []

    for document in documents:

        document_words = tokenize(
            document.page_content
        )

        score = 0

        for q_word in question_words:

            for d_word in document_words:

                # Exact match
                if q_word == d_word:
                    score += 2

                # Simple word-stem matching
                elif (
                    q_word.startswith(d_word)
                    or d_word.startswith(q_word)
                ):
                    score += 1

        scored_documents.append(
            (document, score)
        )

    scored_documents.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        document
        for document, score in scored_documents
    ]