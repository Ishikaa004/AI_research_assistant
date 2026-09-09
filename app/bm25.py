import re
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


def create_bm25_index(documents):

    tokenized_documents = [
        tokenize(document.page_content)
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    return bm25


def bm25_search(bm25, documents, query, k=5):

    tokenized_query = tokenize(query)

    results = bm25.get_top_n(
        tokenized_query,
        documents,
        n=k
    )

    return results