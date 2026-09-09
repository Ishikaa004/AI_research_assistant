def filter_documents(documents, source=None):

    if not source:
        return documents

    filtered_documents = []

    for document in documents:

        document_source = document.metadata.get(
            "source",
            ""
        )

        if source.lower() in document_source.lower():

            filtered_documents.append(
                document
            )

    return filtered_documents