import os
import re

from app.evidence_selector import find_best_evidence

from app.main import (
    initialize_pipeline,
    run_rag,
    llm
)

from app.nli_faithfulness import (
    check_nli_support
)


# ============================================================
# 1. FIND RELEVANT RANKS
# ============================================================

def get_relevant_ranks(
    retrieved_documents,
    expected_source=None,
    expected_page=None
):
    """
    Find the ranks of retrieved chunks that match
    the expected source and page.

    Rank starts from 1.
    """

    if not expected_source:
        return []

    relevant_ranks = []

    for rank, document in enumerate(
        retrieved_documents,
        start=1
    ):
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
            in os.path.basename(source).lower()
        )

        if expected_page is not None:
            page_matches = (
                page == expected_page
            )
        else:
            page_matches = True

        if source_matches and page_matches:
            relevant_ranks.append(
                rank
            )

    return relevant_ranks


# ============================================================
# 2. HIT@K
# ============================================================

def evaluate_hit_at_k(
    retrieved_documents,
    expected_source,
    expected_page=None,
    k=5
):
    """
    Hit@K

    1 -> at least one relevant result
         is present in top K.

    0 -> no relevant result is present
         in top K.
    """

    relevant_ranks = get_relevant_ranks(
        retrieved_documents,
        expected_source,
        expected_page
    )

    for rank in relevant_ranks:

        if rank <= k:
            return 1.0

    return 0.0


# ============================================================
# 3. RECALL@K
# ============================================================

def evaluate_recall_at_k(
    retrieved_documents,
    expected_source,
    expected_page=None,
    k=5
):
    """
    Recall@K for the current evaluation dataset.

    Each test case has one expected relevant page.

    Recall@K = 1 if the expected page appears
    in top K.

    Recall@K = 0 otherwise.
    """

    if not expected_source:
        return 0.0

    for document in retrieved_documents[:k]:

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
            in os.path.basename(source).lower()
        )

        if expected_page is not None:
            page_matches = (
                page == expected_page
            )
        else:
            page_matches = True

        if source_matches and page_matches:
            return 1.0

    return 0.0


# ============================================================
# 4. MRR@K
# ============================================================

def evaluate_mrr_at_k(
    retrieved_documents,
    expected_source,
    expected_page=None,
    k=5
):
    """
    Mean Reciprocal Rank.

    MRR = 1 / rank
    if the relevant result is found.

    Otherwise MRR = 0.
    """

    relevant_ranks = get_relevant_ranks(
        retrieved_documents,
        expected_source,
        expected_page
    )

    for rank in relevant_ranks:

        if rank <= k:
            return 1.0 / rank

    return 0.0


# ============================================================
# 5. BASIC RETRIEVAL SCORE
# ============================================================

def evaluate_retrieval(
    retrieved_documents,
    expected_source,
    expected_page=None
):
    """
    Basic retrieval evaluation.

    Returns:

    1.0 -> expected source/page retrieved
    0.0 -> expected source/page not retrieved
    None -> no expected source
    """

    if not expected_source:
        return None

    relevant_ranks = get_relevant_ranks(
        retrieved_documents,
        expected_source,
        expected_page
    )

    if relevant_ranks:
        return 1.0

    return 0.0


# ============================================================
# 6. ANSWER CORRECTNESS EVALUATION
# ============================================================

def evaluate_answer(
    question,
    answer,
    expected_answer,
    llm
):
    """
    Evaluate answer correctness using an LLM judge.

    The judge compares:

        Question
        Ground truth
        Generated answer

    and returns a score between 0 and 1.

    Returns None when no expected answer exists.
    """

    if expected_answer is None:
        return None

    if not answer:
        return 0.0

    prompt = f"""
You are an evaluator for a Retrieval-Augmented Generation
(RAG) system.

Your task is to evaluate the CORRECTNESS of the generated answer.

You are given:

Question:

{question}

Ground truth answer:

{expected_answer}

Generated answer:

{answer}

Evaluate ONLY whether the generated answer correctly answers
the question compared with the ground truth.

IMPORTANT RULES:

1. Judge the meaning, not exact wording.

2. Accept valid paraphrases.

3. Do not penalize different sentence structure.

4. Do not require the generated answer to use the exact
   words from the ground truth.

5. If the generated answer contains the correct main fact,
   consider it correct even if wording differs.

6. If important information is missing, reduce the score.

7. If the answer contains a factual contradiction,
   reduce the score significantly.

8. If the generated answer contains unsupported extra claims
   that make the answer incorrect, reduce the score.

9. Do NOT use your general knowledge to add facts.

10. Judge only the information provided above.

Scoring:

1.0 = completely correct

0.8 = mostly correct with a minor omission

0.6 = partially correct

0.4 = substantially incomplete or partly incorrect

0.2 = mostly incorrect

0.0 = completely incorrect or does not answer the question

Return ONLY the numeric score.

Examples:

Ground truth:

The Adam optimizer was used.

Generated:

The model was compiled using Adam.

Score:

1.0

Ground truth:

There are 60,000 training images.

Generated:

The training set contains 60,000 images.

Score:

1.0

Ground truth:

The model was saved as mnist_ann_model.keras.

Generated:

The model was saved as model.keras.

Score:

0.0
"""

    try:

        response = llm.invoke(
            prompt
        )

        text = response.content.strip()

        match = re.search(
            r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b",
            text
        )

        if not match:

            print(
                "Answer correctness evaluator "
                "returned an invalid score."
            )

            return 0.0

        score = float(
            match.group()
        )

        score = max(
            0.0,
            min(
                1.0,
                score
            )
        )

        return score

    except Exception as e:

        print(
            f"Answer correctness evaluation error: {e}"
        )

        return 0.0


# ============================================================
# 7. CLAIM-LEVEL FAITHFULNESS
# ============================================================

def extract_claims(
    answer,
    llm
):
    """
    Break the generated answer into individual
    factual claims.

    Returns:

        list -> claims extracted successfully
        []   -> no factual claims
        None -> evaluator failed
    """

    if not answer:
        return []

    prompt = f"""
You are a claim extraction component for a RAG evaluation system.

Extract every factual claim from the generated answer.

Rules:

1. Each claim must represent a factual statement or factual fact.

2. Short factual answers and factual phrases MUST be treated
   as claims.

3. Do not add new information.

4. Do not change the meaning.

5. Ignore greetings and filler text.

6. Return one claim per line.

7. If there are no factual claims, return:

NO_CLAIMS

Examples:

Generated answer:

Adam optimizer.

Output:

Adam optimizer.

Generated answer:

60000 training images.

Output:

There are 60000 training images.

Generated answer:

The MNIST dataset.

Output:

The MNIST dataset.

Generated answer:

Hello, how can I help?

Output:

NO_CLAIMS

Generated answer:

{answer}
"""

    try:

        response = llm.invoke(
            prompt
        )

        text = response.content.strip()

        if text == "NO_CLAIMS":

            # ------------------------------------------------
            # SHORT-ANSWER FALLBACK
            # ------------------------------------------------

            cleaned_answer = answer.strip()

            if cleaned_answer:
                return [
                    cleaned_answer
                ]

            return []

        claims = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            # Remove numbered list formatting
            line = re.sub(
                r"^\s*\d+[\.\)]\s*",
                "",
                line
            )

            # Remove bullet formatting
            line = re.sub(
                r"^\s*[-*]\s*",
                "",
                line
            )

            if line:

                claims.append(
                    line
                )

        # ----------------------------------------------------
        # SHORT-ANSWER FALLBACK
        # ----------------------------------------------------

        if not claims:

            cleaned_answer = answer.strip()

            if cleaned_answer:
                return [
                    cleaned_answer
                ]

        return claims

    except Exception as e:

        print(
            f"Claim extraction error: {e}"
        )

        return None


# ============================================================
# CLAIM SUPPORT CHECK
# ============================================================

def check_claim_support(
    claim,
    context,
    llm
):
    """
    Check whether one factual claim is supported
    by the provided context.

    Returns:

        True  -> supported
        False -> unsupported
        None  -> evaluation failed
    """

    prompt = f"""
You are a faithfulness evaluator for a RAG system.

Your task is to determine whether the CLAIM is directly
supported by the CONTEXT.

CONTEXT:

{context}

CLAIM:

{claim}

Rules:

1. Use ONLY the provided context.

2. Do NOT use outside knowledge.

3. The context may support the claim using different wording.

4. The claim must be directly supported by the context.

5. If the context contradicts the claim, return UNSUPPORTED.

6. If the context does not provide enough information,
   return UNSUPPORTED.

7. Return ONLY one word:

SUPPORTED

or

UNSUPPORTED
"""

    try:

        response = llm.invoke(
            prompt
        )

        result = (
            response.content
            .strip()
            .upper()
        )

        if "UNSUPPORTED" in result:
            return False

        if "SUPPORTED" in result:
            return True

        print(
            "Claim support evaluator returned "
            "an invalid result."
        )

        return None

    except Exception as e:

        print(
            f"Claim support evaluation error: {e}"
        )

        return None


# ============================================================
# LLM-BASED FAITHFULNESS
# ============================================================

def evaluate_faithfulness(
    answer,
    context,
    llm
):
    """
    Evaluate faithfulness at claim level.

    Faithfulness =

        supported claims / evaluated claims

    Returns:

        0.0 to 1.0 -> successful evaluation
        None -> evaluation failed
    """

    if not answer or not context:
        return 0.0

    claims = extract_claims(
        answer,
        llm
    )

    if claims is None:

        print(
            "Faithfulness evaluation: N/A "
            "(claim extraction failed)"
        )

        return None

    if not claims:

        print(
            "Faithfulness evaluation: 0.0 "
            "(no factual claims)"
        )

        return 0.0

    print(
        f"\nClaims extracted: {len(claims)}"
    )

    supported_claims = 0
    evaluated_claims = 0

    for index, claim in enumerate(
        claims,
        start=1
    ):

        supported = check_claim_support(
            claim,
            context,
            llm
        )

        if supported is True:

            supported_claims += 1
            evaluated_claims += 1

            print(
                f"Claim {index}: SUPPORTED"
            )

        elif supported is False:

            evaluated_claims += 1

            print(
                f"Claim {index}: UNSUPPORTED"
            )

        else:

            print(
                f"Claim {index}: N/A "
                "(evaluation failed)"
            )

        print(
            f"  {claim}"
        )

    if evaluated_claims == 0:

        print(
            "Faithfulness evaluation: N/A "
            "(no claims could be evaluated)"
        )

        return None

    score = (
        supported_claims
        / evaluated_claims
    )

    print(
        f"Supported claims: "
        f"{supported_claims}/{evaluated_claims}"
    )

    return score


# ============================================================
# 8. NLI-BASED FAITHFULNESS
# ============================================================

def evaluate_nli_faithfulness(
    answer,
    context,
    retrieved_documents,
    llm
):
    """
    Evaluate faithfulness using NLI.

    Each factual claim is first matched with
    its most relevant retrieved document chunk.

    Then NLI checks the claim against that
    specific evidence.

    Returns:

        0.0 to 1.0 -> successful evaluation
        None -> evaluation failed
    """

    if not answer or not context:
        return 0.0

    # --------------------------------------------------------
    # Extract claims
    # --------------------------------------------------------

    claims = extract_claims(
        answer,
        llm
    )

    if claims is None:

        print(
            "NLI faithfulness: N/A "
            "(claim extraction failed)"
        )

        return None

    if not claims:

        print(
            "NLI faithfulness: 0.0 "
            "(no factual claims)"
        )

        return 0.0

    print(
        f"\nNLI claims: {len(claims)}"
    )

    supported_claims = 0
    evaluated_claims = 0

    # --------------------------------------------------------
    # Evaluate each claim
    # --------------------------------------------------------

    for index, claim in enumerate(
        claims,
        start=1
    ):

        print(
            f"\nNLI Claim {index}:"
        )

        print(
            f"  {claim}"
        )

        try:

            # ------------------------------------------------
            # Find best evidence for this claim
            # ------------------------------------------------

            evidence_document = find_best_evidence(
                claim,
                retrieved_documents
            )

            if evidence_document is None:

                print(
                    "  No relevant evidence found."
                )

                continue

            # ------------------------------------------------
            # Get evidence text
            # ------------------------------------------------

            evidence = (
                evidence_document.page_content
            )

            print(
                "  Evidence:"
            )

            print(
                f"  {evidence}"
            )

            # ------------------------------------------------
            # NLI check
            # ------------------------------------------------

            result = check_nli_support(
                claim,
                evidence
            )

            print(
                f"  NLI Result: {result}"
            )

            # ------------------------------------------------
            # Score
            # ------------------------------------------------

            if result == "ENTAILMENT":

                supported_claims += 1
                evaluated_claims += 1

            elif result in [
                "CONTRADICTION",
                "NEUTRAL"
            ]:

                evaluated_claims += 1

            else:

                print(
                    "  Invalid NLI result."
                )

        except Exception as e:

            print(
                f"NLI evaluation error: {e}"
            )

    # --------------------------------------------------------
    # No claims evaluated
    # --------------------------------------------------------

    if evaluated_claims == 0:

        print(
            "NLI faithfulness: N/A "
            "(no claims evaluated)"
        )

        return None

    # --------------------------------------------------------
    # Calculate score
    # --------------------------------------------------------

    score = (
        supported_claims
        / evaluated_claims
    )

    print(
        f"\nNLI supported claims: "
        f"{supported_claims}/{evaluated_claims}"
    )

    print(
        f"NLI faithfulness: {score:.2f}"
    )

    return score


# ============================================================
# 9. UNSUPPORTED QUESTION EVALUATION
# ============================================================

def evaluate_unsupported_answer(
    answer,
    expected_unsupported=True
):
    """
    Check whether the system correctly refuses
    unsupported questions.
    """

    expected_message = (
        "I don't know based on the provided documents."
    )

    actual_unsupported = (
        expected_message.lower()
        in answer.lower()
    )

    if expected_unsupported:

        if actual_unsupported:
            return 1.0

        return 0.0

    if not actual_unsupported:
        return 1.0

    return 0.0


# ============================================================
# 10. RUN EVALUATION
# ============================================================

def run_evaluation(
    test_cases,
    document_path
):
    """
    Run complete RAG evaluation.

    Metrics:

    Retrieval:
        - Hit@5
        - Recall@5
        - MRR@5
        - Basic retrieval score

    Generation:
        - Answer correctness

    Faithfulness:
        - Claim-level LLM faithfulness
        - NLI-based faithfulness

    Safety:
        - Unsupported handling
    """

    print(
        "\n========================================"
    )

    print(
        "RAG EVALUATION"
    )

    print(
        "========================================"
    )

    print(
        "\nLoading evaluation document:"
    )

    print(
        document_path
    )

    # ========================================================
    # LOAD DOCUMENT
    # ========================================================

    initialize_pipeline(
        [document_path]
    )

    print(
        "\nEvaluation document loaded."
    )

    # ========================================================
    # METRIC LISTS
    # ========================================================

    retrieval_scores = []
    hit_scores = []
    recall_scores = []
    mrr_scores = []

    answer_scores = []

    faithfulness_scores = []
    nli_faithfulness_scores = []

    unsupported_scores = []

    # ========================================================
    # RUN TEST CASES
    # ========================================================

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

        expected_unsupported = test_case.get(
            "expected_unsupported",
            False
        )

        print(
            "\n========================================"
        )

        print(
            f"TEST {index}"
        )

        print(
            "========================================"
        )

        print(
            "\nQuestion:",
            question
        )

        # ====================================================
        # RUN RAG
        # ====================================================

        answer, retrieved_documents, context = run_rag(
            question=question,
            use_memory=False,
            save_memory=False,
            verbose=False
        )

        print(
            "\nAnswer:",
            answer
        )

        # ====================================================
        # RETRIEVAL METRICS
        # ====================================================

        if expected_source:

            hit_at_5 = evaluate_hit_at_k(
                retrieved_documents,
                expected_source,
                expected_page,
                k=5
            )

            recall_at_5 = evaluate_recall_at_k(
                retrieved_documents,
                expected_source,
                expected_page,
                k=5
            )

            mrr_at_5 = evaluate_mrr_at_k(
                retrieved_documents,
                expected_source,
                expected_page,
                k=5
            )

            retrieval_score = evaluate_retrieval(
                retrieved_documents,
                expected_source,
                expected_page
            )

            hit_scores.append(
                hit_at_5
            )

            recall_scores.append(
                recall_at_5
            )

            mrr_scores.append(
                mrr_at_5
            )

            retrieval_scores.append(
                retrieval_score
            )

            print(
                "\nHit@5:",
                hit_at_5
            )

            print(
                "Recall@5:",
                recall_at_5
            )

            print(
                "MRR@5:",
                round(
                    mrr_at_5,
                    3
                )
            )

            print(
                "Basic retrieval score:",
                retrieval_score
            )

        else:

            print(
                "\nHit@5: N/A"
            )

            print(
                "Recall@5: N/A"
            )

            print(
                "MRR@5: N/A"
            )

            print(
                "Basic retrieval score: N/A"
            )

        # ====================================================
        # ANSWER CORRECTNESS
        # ====================================================

        answer_score = evaluate_answer(
            question,
            answer,
            expected_answer,
            llm
        )

        if answer_score is not None:

            answer_scores.append(
                answer_score
            )

            print(
                "\nAnswer correctness:",
                round(
                    answer_score,
                    2
                )
            )

        else:

            print(
                "\nAnswer correctness: N/A"
            )

        # ====================================================
        # UNSUPPORTED HANDLING
        # ====================================================

        if expected_unsupported:

            unsupported_score = (
                evaluate_unsupported_answer(
                    answer,
                    expected_unsupported=True
                )
            )

            unsupported_scores.append(
                unsupported_score
            )

            print(
                "Unsupported handling:",
                unsupported_score
            )

        # ====================================================
        # CLAIM-LEVEL LLM FAITHFULNESS
        # ====================================================

        if expected_unsupported:

            print(
                "Faithfulness score: N/A "
                "(unsupported question)"
            )

        else:

            faithfulness_score = (
                evaluate_faithfulness(
                    answer,
                    context,
                    llm
                )
            )

            if faithfulness_score is not None:

                faithfulness_scores.append(
                    faithfulness_score
                )

                print(
                    "Faithfulness score:",
                    round(
                        faithfulness_score,
                        2
                    )
                )

            else:

                print(
                    "Faithfulness score: N/A "
                    "(evaluation failed)"
                )

        # ====================================================
        # NLI-BASED FAITHFULNESS
        # ====================================================

        if expected_unsupported:

            print(
                "NLI Faithfulness: N/A "
                "(unsupported question)"
            )

        else:

            # IMPORTANT:
            # Pass retrieved_documents here.
            # This fixes the previous TypeError.

            nli_faithfulness_score = (
                evaluate_nli_faithfulness(
                    answer,
                    context,
                    retrieved_documents,
                    llm
                )
            )

            if nli_faithfulness_score is not None:

                nli_faithfulness_scores.append(
                    nli_faithfulness_score
                )

                print(
                    "NLI Faithfulness score:",
                    round(
                        nli_faithfulness_score,
                        2
                    )
                )

            else:

                print(
                    "NLI Faithfulness: N/A "
                    "(evaluation failed)"
                )

        # ====================================================
        # RETRIEVED SOURCES
        # ====================================================

        print(
            "\nRetrieved sources:"
        )

        seen_sources = set()

        for document in retrieved_documents:

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                None
            )

            source_name = os.path.basename(
                source
            )

            source_key = (
                source_name,
                page
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(
                source_key
            )

            if page is not None:

                print(
                    f"- {source_name} "
                    f"(Page {page})"
                )

            else:

                print(
                    f"- {source_name}"
                )

        if not retrieved_documents:

            print(
                "- No supporting source found."
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print(
        "\n\n========================================"
    )

    print(
        "FINAL EVALUATION RESULTS"
    )

    print(
        "========================================"
    )

    # ========================================================
    # BASIC RETRIEVAL
    # ========================================================

    if retrieval_scores:

        retrieval_average = (
            sum(retrieval_scores)
            / len(retrieval_scores)
        )

        print(
            "\nRetrieval average:",
            round(
                retrieval_average,
                2
            )
        )

    # ========================================================
    # HIT@5
    # ========================================================

    if hit_scores:

        hit_average = (
            sum(hit_scores)
            / len(hit_scores)
        )

        print(
            "Hit@5:",
            round(
                hit_average,
                2
            )
        )

    # ========================================================
    # RECALL@5
    # ========================================================

    if recall_scores:

        recall_average = (
            sum(recall_scores)
            / len(recall_scores)
        )

        print(
            "Recall@5:",
            round(
                recall_average,
                2
            )
        )

    # ========================================================
    # MRR@5
    # ========================================================

    if mrr_scores:

        mrr_average = (
            sum(mrr_scores)
            / len(mrr_scores)
        )

        print(
            "MRR@5:",
            round(
                mrr_average,
                2
            )
        )

    # ========================================================
    # ANSWER CORRECTNESS
    # ========================================================

    if answer_scores:

        answer_average = (
            sum(answer_scores)
            / len(answer_scores)
        )

        print(
            "Answer correctness average:",
            round(
                answer_average,
                2
            )
        )

    # ========================================================
    # UNSUPPORTED HANDLING
    # ========================================================

    if unsupported_scores:

        unsupported_average = (
            sum(unsupported_scores)
            / len(unsupported_scores)
        )

        print(
            "Unsupported handling:",
            round(
                unsupported_average,
                2
            )
        )

    # ========================================================
    # CLAIM-LEVEL FAITHFULNESS
    # ========================================================

    if faithfulness_scores:

        faithfulness_average = (
            sum(faithfulness_scores)
            / len(faithfulness_scores)
        )

        print(
            "Claim-level faithfulness average:",
            round(
                faithfulness_average,
                2
            )
        )

    else:

        print(
            "Claim-level faithfulness average: N/A"
        )

    # ========================================================
    # NLI FAITHFULNESS
    # ========================================================

    if nli_faithfulness_scores:

        nli_faithfulness_average = (
            sum(nli_faithfulness_scores)
            / len(nli_faithfulness_scores)
        )

        print(
            "NLI faithfulness average:",
            round(
                nli_faithfulness_average,
                2
            )
        )

    else:

        print(
            "NLI faithfulness average: N/A"
        )

    print(
        "\nEvaluation completed."
    )


# ============================================================
# 11. TEST DATASET
# ============================================================

if __name__ == "__main__":

    document_path = (
        r"C:\Users\Dell\Downloads\AI_research_assistant"
        r"\data\MNIST_ANN_Project_Notes.pdf"
    )

    test_cases = [

        # ----------------------------------------------------
        # TEST 1
        # ----------------------------------------------------

        {
            "question":
                "What dataset was used in this project?",

            "expected_source":
                "MNIST_ANN_Project_Notes.pdf",

            "expected_page":
                1,

            "expected_answer":
                "The MNIST dataset was used."
        },

        # ----------------------------------------------------
        # TEST 2
        # ----------------------------------------------------

        {
            "question":
                "How many training images are in the MNIST dataset?",

            "expected_source":
                "MNIST_ANN_Project_Notes.pdf",

            "expected_page":
                1,

            "expected_answer":
                "There are 60,000 training images."
        },

        # ----------------------------------------------------
        # TEST 3
        # ----------------------------------------------------

        {
            "question":
                "How many test images are in the MNIST dataset?",

            "expected_source":
                "MNIST_ANN_Project_Notes.pdf",

            "expected_page":
                1,

            "expected_answer":
                "There are 10,000 test images."
        },

        # ----------------------------------------------------
        # TEST 4
        # ----------------------------------------------------

        {
            "question":
                "Why were the pixel values divided by 255?",

            "expected_source":
                "MNIST_ANN_Project_Notes.pdf",

            "expected_page":
                1,

            "expected_answer":
                "The pixel values were divided by 255 to normalize them to the range 0 to 1, which improves convergence and training stability."
        },

        # ----------------------------------------------------
        # TEST 5
        # ----------------------------------------------------

        {
            "question":
                "What is the architecture of the ANN?",

            "expected_source":
                "MNIST_ANN_Project_Notes.pdf",

            "expected_page":
                1,

            "expected_answer":
                "The ANN uses a Flatten layer, a Dense layer with 128 neurons and ReLU activation, and a Dense output layer with 10 neurons and Softmax activation."
        },

        # ----------------------------------------------------
        # TEST 6
        # ----------------------------------------------------

        {
            "question":
                "What optimizer was used to compile the model?",

            "expected_source":
                "MNIST_ANN_Project_Notes.pdf",

            "expected_page":
                2,

            "expected_answer":
                "The Adam optimizer was used."
        },

        # ----------------------------------------------------
        # TEST 7
        # ----------------------------------------------------

        {
            "question":
                "What loss function was used?",

            "expected_source":
                "MNIST_ANN_Project_Notes.pdf",

            "expected_page":
                2,

            "expected_answer":
                "Sparse categorical crossentropy was used."
        },

        # ----------------------------------------------------
        # TEST 8
        # ----------------------------------------------------

        {
            "question":
                "How was the ANN trained?",

            "expected_source":
                "MNIST_ANN_Project_Notes.pdf",

            "expected_page":
                2,

            "expected_answer":
                "The ANN was trained using model.fit with epochs=10, batch_size=32, and validation_split=0.2."
        },

        # ----------------------------------------------------
        # TEST 9
        # ----------------------------------------------------

        {
            "question":
                "How were predictions converted into class labels?",

            "expected_source":
                "MNIST_ANN_Project_Notes.pdf",

            "expected_page":
                2,

            "expected_answer":
                "The argmax operation was used to obtain the predicted class label."
        },

        # ----------------------------------------------------
        # TEST 10
        # ----------------------------------------------------

        {
            "question":
                "What filename was used to save the trained model?",

            "expected_source":
                "MNIST_ANN_Project_Notes.pdf",

            "expected_page":
                3,

            "expected_answer":
                "The trained model was saved as mnist_ann_model.keras."
        },

        # ----------------------------------------------------
        # TEST 11 - NEGATIVE TEST
        # ----------------------------------------------------

        {
            "question":
                "What accuracy did the ANN achieve on the test set?",

            "expected_source":
                None,

            "expected_page":
                None,

            "expected_answer":
                None,

            "expected_unsupported":
                True
        }
    ]

    # ========================================================
    # RUN
    # ========================================================

    run_evaluation(
        test_cases,
        document_path
    )