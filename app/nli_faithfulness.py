from transformers import pipeline


# Load the NLI model once
nli_model = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-base",
    top_k=None
)


def check_nli_support(
    claim,
    evidence
):
    """
    Check whether one evidence passage supports one claim.

    Returns:

        ENTAILMENT
        CONTRADICTION
        NEUTRAL
    """

    result = nli_model(
        {
            "text": evidence,
            "text_pair": claim
        }
    )

    # Transformers may return:
    #
    # [[{...}, {...}, {...}]]
    #
    # or:
    #
    # [{...}, {...}, {...}]
    if (
        isinstance(result, list)
        and len(result) > 0
        and isinstance(result[0], list)
    ):
        scores = result[0]
    else:
        scores = result

    best_result = max(
        scores,
        key=lambda x: x["score"]
    )

    return best_result["label"].upper()