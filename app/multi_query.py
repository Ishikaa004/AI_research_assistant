from langchain_core.prompts import ChatPromptTemplate


def generate_queries(llm, question):

    prompt = ChatPromptTemplate.from_template(
        """
        Generate 3 different search queries for the user's question.

        Each query should express the same information need
        using different wording.

        Return only the 3 queries, one per line.

        User question:
        {question}
        """
    )

    chain = prompt | llm

    response = chain.invoke({
        "question": question
    })

    queries = response.content.strip().split("\n")

    queries = [
        query.strip("0123456789.- ")
        for query in queries
        if query.strip()
    ]

    return queries[:3]