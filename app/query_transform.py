from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def transform_query(
    llm,
    question,
    chat_history=""
):
    """
    Safely transform the user's question.

    The goal is to improve clarity without changing
    the user's actual topic or meaning.
    """

    question = question.strip()

    # --------------------------------------------------------
    # If the question is already reasonably clear,
    # keep it unchanged.
    # --------------------------------------------------------

    if len(question.split()) >= 3:

        return question


    # --------------------------------------------------------
    # Only use the LLM for very short / ambiguous questions.
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
You are a query rewriting component in a RAG system.

Rewrite the user's question into a standalone search query.

STRICT RULES:

1. Do NOT answer the question.

2. Do NOT introduce a new topic.

3. Do NOT assume the topic is GenAI, AWS,
   machine learning, transformers, or anything else.

4. Preserve the exact meaning of the user's question.

5. Use conversation history ONLY to resolve
   pronouns or references such as:
   it, they, this, that, these, those.

6. If you cannot determine what a reference means,
   keep the original wording.

7. Never invent a topic.

8. Return ONLY the rewritten question.

Conversation history:
{chat_history}

User question:
{question}
"""
        ),

        (
            "human",
            "{question}"
        )
    ])

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    transformed_query = chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    return transformed_query.strip()