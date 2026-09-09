def reciprocal_rank_fusion(
    result_lists,
    k=60
):
    """
    Combine multiple ranked document lists
    using Reciprocal Rank Fusion.
    """

    scores = {}
    documents = {}

    for result_list in result_lists:

        for rank, document in enumerate(
            result_list,
            start=1
        ):

            source = document.metadata.get(
                "source",
                ""
            )

            content = document.page_content

            doc_id = source + content

            if doc_id not in scores:
                scores[doc_id] = 0
                documents[doc_id] = document

            scores[doc_id] += 1 / (
                k + rank
            )

    ranked_documents = sorted(
        scores.keys(),
        key=lambda x: scores[x],
        reverse=True
    )

    return [
        documents[doc_id]
        for doc_id in ranked_documents
    ]