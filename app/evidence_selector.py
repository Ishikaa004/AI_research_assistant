import re

from langchain_core.documents import Document


# =========================================================
# STOP WORDS
# =========================================================

STOP_WORDS = {
    "the", "a", "an",
    "is", "are", "was", "were",
    "what", "how", "why", "when",
    "where", "which",
    "and", "or", "of", "to",
    "in", "on", "for", "with",
    "did", "do", "does",
    "this", "that", "these", "those",
    "it", "they", "them",
    "be", "been", "being",
    "by", "from", "as",
    "into", "using"
}


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(text):
    words = re.findall(
        r"\b[a-zA-Z0-9_.-]+\b",
        text.lower()
    )

    return [
        word
        for word in words
        if word not in STOP_WORDS
    ]


def normalize(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_numbers(text):
    return set(
        re.findall(
            r"\b\d+(?:\.\d+)?\b",
            text
        )
    )


def extract_special_tokens(text):
    tokens = re.findall(
        r"[a-zA-Z_][a-zA-Z0-9_.-]*",
        text.lower()
    )

    return set(tokens)


# =========================================================
# STRUCTURAL SPLITTING
# =========================================================

def split_into_units(text):
    """
    Split PDF text into small structural evidence units.

    Examples:

        Step 2: Load Dataset
        What did I do? Loaded MNIST dataset...
        Why? Training data teaches...
        Output: 60000 train images...

    Each of these becomes a separate evidence unit.
    """

    # Normalize line breaks
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Add line breaks before structural markers
    markers = [
        r"Step\s+\d+\s*:",
        r"What did I do\?",
        r"Why\?",
        r"Output\s*:"
    ]

    for marker in markers:
        text = re.sub(
            rf"\s+(?={marker})",
            "\n",
            text,
            flags=re.IGNORECASE
        )

    raw_lines = text.split("\n")

    units = []

    for line in raw_lines:

        line = line.strip()

        if not line:
            continue

        # Split very long normal lines into sentences
        if len(line) > 220:

            sentences = re.split(
                r"(?<=[.!?])\s+",
                line
            )

            for sentence in sentences:

                sentence = sentence.strip()

                if sentence:
                    units.append(sentence)

        else:
            units.append(line)

    return units


# =========================================================
# QUESTION TYPE
# =========================================================

def is_simple_claim(claim):
    """
    Detect short atomic claims such as:

        MNIST dataset.
        60000 training images.
        Adam optimizer.
        mnist_ann_model.keras.
    """

    words = tokenize(claim)

    return len(words) <= 6


# =========================================================
# EVIDENCE SCORING
# =========================================================

def score_evidence(claim, evidence):

    claim_normalized = normalize(claim)
    evidence_normalized = normalize(evidence)

    score = 0

    # -----------------------------------------------------
    # Exact phrase match
    # -----------------------------------------------------

    if (
        claim_normalized
        and claim_normalized in evidence_normalized
    ):
        score += 500

    # -----------------------------------------------------
    # Token overlap
    # -----------------------------------------------------

    claim_tokens = set(tokenize(claim))
    evidence_tokens = set(tokenize(evidence))

    overlap = claim_tokens & evidence_tokens

    score += len(overlap) * 15

    # -----------------------------------------------------
    # Numbers are very important
    # -----------------------------------------------------

    claim_numbers = extract_numbers(claim)
    evidence_numbers = extract_numbers(evidence)

    number_overlap = (
        claim_numbers & evidence_numbers
    )

    score += len(number_overlap) * 100

    # -----------------------------------------------------
    # Technical tokens
    # -----------------------------------------------------

    claim_special = extract_special_tokens(claim)
    evidence_special = extract_special_tokens(evidence)

    special_overlap = (
        claim_special & evidence_special
    )

    score += len(special_overlap) * 30

    # -----------------------------------------------------
    # Technical phrases
    # -----------------------------------------------------

    technical_phrases = [
        "mnist",
        "adam",
        "optimizer",
        "loss",
        "sparse categorical crossentropy",
        "model.fit",
        "batch_size",
        "validation_split",
        "epochs",
        "argmax",
        "softmax",
        "relu",
        "flatten",
        "dense",
        "mnist_ann_model.keras",
        "train images",
        "training images",
        "test images",
        "pixel values",
        "convergence",
        "training stability"
    ]

    for phrase in technical_phrases:

        if phrase in claim_normalized:

            if phrase in evidence_normalized:
                score += 80

    # -----------------------------------------------------
    # Length penalty
    #
    # Prefer smaller evidence.
    # -----------------------------------------------------

    score -= len(evidence) * 0.15

    return score


# =========================================================
# FIND BEST EVIDENCE
# =========================================================

def find_best_evidence(claim, documents):

    """
    Find the smallest relevant evidence.

    For simple claims:
        return one evidence unit.

    For composite claims:
        also consider 2-3 adjacent evidence units.
    """

    if not documents:
        return None

    all_units = []

    # -----------------------------------------------------
    # Create evidence units
    # -----------------------------------------------------

    for document in documents:

        units = split_into_units(
            document.page_content
        )

        for index, unit in enumerate(units):

            all_units.append(
                {
                    "document": document,
                    "unit": unit,
                    "index": index
                }
            )

    if not all_units:
        return None

    # -----------------------------------------------------
    # SIMPLE CLAIM
    # -----------------------------------------------------

    if is_simple_claim(claim):

        best = None
        best_score = float("-inf")

        for item in all_units:

            score = score_evidence(
                claim,
                item["unit"]
            )

            if score > best_score:

                best_score = score
                best = item

        if best is None:
            return None

        selected_document = Document(
            page_content=best["unit"],
            metadata=best["document"].metadata
        )

        print("\nEvidence selection:")
        print(
            f"  Selected evidence score: "
            f"{best_score:.2f}"
        )
        print(
            f"  Evidence:\n"
            f"  {best['unit']}"
        )

        return selected_document

    # -----------------------------------------------------
    # COMPOSITE CLAIM
    # -----------------------------------------------------
    #
    # Example:
    #
    # "The model was trained using model.fit
    # with 10 epochs and batch size 32."
    #
    # One line may not contain everything.
    # Therefore check windows of 2-3 units.
    # -----------------------------------------------------

    candidates = []

    for item in all_units:

        index = item["index"]
        document = item["document"]

        # Find units belonging to the same document
        document_units = split_into_units(
            document.page_content
        )

        for window_size in [2, 3]:

            if index + window_size > len(document_units):
                continue

            window = document_units[
                index:index + window_size
            ]

            combined = " ".join(window)

            score = score_evidence(
                claim,
                combined
            )

            # Prefer smaller windows
            score -= window_size * 5

            candidates.append(
                {
                    "document": document,
                    "text": "\n".join(window),
                    "score": score
                }
            )

    # -----------------------------------------------------
    # Select best candidate
    # -----------------------------------------------------

    if candidates:

        best = max(
            candidates,
            key=lambda x: x["score"]
        )

        selected_document = Document(
            page_content=best["text"],
            metadata=best["document"].metadata
        )

        print("\nEvidence selection:")
        print(
            f"  Selected evidence score: "
            f"{best['score']:.2f}"
        )
        print(
            f"  Evidence:\n"
            f"  {best['text']}"
        )

        return selected_document

    return None