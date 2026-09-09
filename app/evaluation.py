from app.main import run_rag, initialize_pipeline, llm


# ============================================================
# 1. RETRIEVAL EVALUATION
# ============================================================

def evaluate_retrieval(
    retrieved_documents,
    expected_source,
    expected_page=None
):
    """
    Check whether the expected source was retrieved.

    If expected_page is provided, the document must also
    contain that page.
    """

    if expected_source is None:
        return 0

    for document in retrieved_documents:

        source = document.metadata.get(
            "source",
            ""
        )

        page = document.metadata.get(
            "page",
            None
        )

        source_matches = (
            expected_source.lower()
            in source.lower()
        )

        if expected_page is not None:

            page_matches = (
                page == expected_page
            )

            if source_matches and page_matches:
                return 1

        else:

            if source_matches:
                return 1

    return 0


# ============================================================
# 2. ANSWER CORRECTNESS EVALUATION
# ============================================================

def evaluate_answer(
    answer,
    expected_answer
):
    """
    Simple keyword-based answer evaluation.

    Returns a score between 0 and 1.

    For this project this is a lightweight baseline metric.
    """

    if expected_answer is None:
        return None

    if not answer:
        return 0

    answer = answer.lower()
    expected_answer = expected_answer.lower()

    keywords = expected_answer.split()

    if not keywords:
        return 0

    matched = 0

    for keyword in keywords:

        if keyword in answer:
            matched += 1

    score = matched / len(keywords)

    return score


# ============================================================
# 3. LLM-BASED FAITHFULNESS EVALUATION
# ============================================================

def evaluate_faithfulness(
    answer,
    context
):
    """
    Use the LLM to determine whether the answer is
    supported by the retrieved context.

    Returns a score between 0 and 1.
    """

    if not answer or not context:
        return 0

    prompt = f"""
You are evaluating a RAG system.

Your task is to determine whether the answer is
supported ONLY by the provided context.

CONTEXT:
{context}

ANSWER:
{answer}

RULES:

1. Every factual claim in the answer must be
   supported by the context.

2. Do not use outside knowledge.

3. If the answer contains information that is
   not supported by the context, reduce the score.

4. If the answer is completely supported,
   give a score of 1.

5. If the answer is partially supported,
   give a score between 0 and 1.

6. If the answer is completely unsupported,
   give a score of 0.

7. If the answer says:
   "I don't know based on the provided documents."
   and the context does not contain the requested
   information, consider that answer faithful.

Return ONLY a number between 0 and 1.
"""

    try:

        response = llm.invoke(
            prompt
        )

        score_text = response.content.strip()

        try:

            score = float(
                score_text
            )

        except ValueError:

            import re

            match = re.search(
                r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b",
                score_text
            )

            if match:

                score = float(
                    match.group()
                )

            else:

                score = 0

        # Keep score between 0 and 1.

        score = max(
            0,
            min(
                1,
                score
            )
        )

        return score

    except Exception as e:

        print(
            f"Faithfulness evaluation error: {e}"
        )

        return 0


# ============================================================
# 4. UNSUPPORTED QUESTION EVALUATION
# ============================================================

def evaluate_unsupported_answer(
    answer,
    expected_answer
):
    """
    Check whether an unsupported question correctly
    produces the required fallback response.
    """

    if expected_answer is None:
        return 0

    required_response = (
        "I don't know based on the provided documents."
    )

    if answer.strip() == required_response:
        return 1

    return 0


# ============================================================
# 5. RUN EVALUATION
# ============================================================

def run_evaluation(
    test_cases,
    rag_function
):
    """
    Run evaluation over multiple question-answer
    test cases.
    """

    results = []

    for index, test_case in enumerate(
        test_cases,
        start=1
    ):

        question = test_case[
            "question"
        ]

        expected_source = test_case.get(
            "expected_source"
        )

        expected_page = test_case.get(
            "expected_page"
        )

        expected_answer = test_case.get(
            "expected_answer"
        )

        print(
            "\n========================================"
        )

        print(
            f"TEST CASE {index}"
        )

        print(
            "========================================"
        )

        print(
            "Question:",
            question
        )

        # ----------------------------------------------------
        # RUN RAG
        # ----------------------------------------------------

        answer, documents, context = rag_function(
            question=question,
            use_memory=False,
            save_memory=False,
            verbose=False
        )

        print()

        print(
            "Answer:",
            answer
        )

        # ----------------------------------------------------
        # RETRIEVAL SCORE
        # ----------------------------------------------------

        if expected_source is not None:

            retrieval_score = evaluate_retrieval(
                documents,
                expected_source,
                expected_page
            )

        else:

            retrieval_score = None

        # ----------------------------------------------------
        # ANSWER SCORE
        # ----------------------------------------------------

        if expected_answer is not None:

            answer_score = evaluate_answer(
                answer,
                expected_answer
            )

        else:

            answer_score = None

        # ----------------------------------------------------
        # UNSUPPORTED QUESTION
        # ----------------------------------------------------

        unsupported_score = None

        if (
            expected_source is None
            and expected_answer is None
        ):

            unsupported_score = (
                evaluate_unsupported_answer(
                    answer,
                    "unsupported"
                )
            )

        # ----------------------------------------------------
        # FAITHFULNESS
        # ----------------------------------------------------

        faithfulness_score = evaluate_faithfulness(
            answer,
            context
        )

        # ----------------------------------------------------
        # DISPLAY SCORES
        # ----------------------------------------------------

        if retrieval_score is not None:

            print(
                "Retrieval score:",
                retrieval_score
            )

        else:

            print(
                "Retrieval score: N/A"
            )

        if answer_score is not None:

            print(
                "Answer score:",
                round(
                    answer_score,
                    2
                )
            )

        else:

            print(
                "Answer score: N/A"
            )

        if unsupported_score is not None:

            print(
                "Unsupported handling:",
                unsupported_score
            )

        print(
            "Faithfulness score:",
            round(
                faithfulness_score,
                2
            )
        )

        # ----------------------------------------------------
        # DISPLAY SOURCES
        # ----------------------------------------------------

        print(
            "\nRetrieved sources:"
        )

        if documents:

            seen_sources = set()

            for document in documents:

                source = document.metadata.get(
                    "source",
                    "Unknown"
                )

                page = document.metadata.get(
                    "page",
                    None
                )

                source_name = source

                source_key = (
                    source_name,
                    page
                )

                if source_key in seen_sources:
                    continue

                seen_sources.add(
                    source_key
                )

                if page:

                    print(
                        f"- {source_name} "
                        f"(Page {page})"
                    )

                else:

                    print(
                        f"- {source_name}"
                    )

        else:

            print(
                "- No supporting source found."
            )

        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        results.append({

            "question":
                question,

            "retrieval_score":
                retrieval_score,

            "answer_score":
                answer_score,

            "unsupported_score":
                unsupported_score,

            "faithfulness_score":
                faithfulness_score
        })

    return results


# ============================================================
# 6. TEST DATASET
# ============================================================

test_cases = [

    {
        "question":
            "What is data engineering?",

        "expected_source":
            "Data Engineering -1.pdf",

        "expected_page":
            None,

        "expected_answer":
            "data engineering"
    },

    {
        "question":
            "What AWS services are mentioned?",

        "expected_source":
            "Data Engineering -1.pdf",

        "expected_page":
            None,

        "expected_answer":
            "AWS Lambda AWS Glue"
    },

    {
        "question":
            "What technologies are discussed in the document?",

        "expected_source":
            "Data Engineering -1.pdf",

        "expected_page":
            None,

        "expected_answer":
            "AWS Lambda AWS Glue Parquet"
    },

    # --------------------------------------------------------
    # NEGATIVE TEST
    # --------------------------------------------------------

    {
        "question":
            "What accuracy did the model achieve?",

        "expected_source":
            None,

        "expected_page":
            None,

        "expected_answer":
            None
    }
]


# ============================================================
# 7. MAIN EVALUATION
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # EVALUATION DOCUMENT
    # --------------------------------------------------------

    document_path = (
        r"C:\Users\Dell\Downloads\AI_research_assistant"
        r"\data\Data Engineering -1.pdf"
    )

    print(
        "\n========================================"
    )

    print(
        "INITIALIZING EVALUATION DOCUMENT"
    )

    print(
        "========================================"
    )

    try:

        initialize_pipeline([
            document_path
        ])

    except ValueError as e:

        print(
            f"\nError loading evaluation document: {e}"
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # RUN EVALUATION
    # --------------------------------------------------------

    results = run_evaluation(
        test_cases,
        run_rag
    )

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    retrieval_scores = [
        result["retrieval_score"]
        for result in results
        if result["retrieval_score"] is not None
    ]

    answer_scores = [
        result["answer_score"]
        for result in results
        if result["answer_score"] is not None
    ]

    unsupported_scores = [
        result["unsupported_score"]
        for result in results
        if result["unsupported_score"] is not None
    ]

    faithfulness_scores = [
        result["faithfulness_score"]
        for result in results
    ]

    # --------------------------------------------------------
    # AVERAGES
    # --------------------------------------------------------

    retrieval_average = (
        sum(retrieval_scores)
        / len(retrieval_scores)
        if retrieval_scores
        else 0
    )

    answer_average = (
        sum(answer_scores)
        / len(answer_scores)
        if answer_scores
        else 0
    )

    unsupported_average = (
        sum(unsupported_scores)
        / len(unsupported_scores)
        if unsupported_scores
        else 0
    )

    faithfulness_average = (
        sum(faithfulness_scores)
        / len(faithfulness_scores)
        if faithfulness_scores
        else 0
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "FINAL EVALUATION SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        "Retrieval average:",
        round(
            retrieval_average,
            2
        )
    )

    print(
        "Answer average:",
        round(
            answer_average,
            2
        )
    )

    print(
        "Unsupported handling:",
        round(
            unsupported_average,
            2
        )
    )

    print(
        "Faithfulness average:",
        round(
            faithfulness_average,
            2
        )
    )

    print(
        "\nEvaluation completed."
    )

