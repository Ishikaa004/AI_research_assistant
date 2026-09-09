import os

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import pymupdf


def load_document(file_path):
    """
    Load a PDF or TXT document.

    Returns:
        List[Document]
    """

    file_path = os.path.abspath(file_path)

    filename = os.path.basename(file_path)

    # =========================================================
    # PDF
    # =========================================================

    if file_path.lower().endswith(".pdf"):

        try:

            pdf = pymupdf.open(file_path)

            documents = []

            for page_number, page in enumerate(pdf):

                try:

                    text = page.get_text()

                    if text.strip():

                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": filename,
                                    "page": page_number + 1
                                }
                            )
                        )

                except Exception as e:

                    print(
                        f"Warning: Could not read page "
                        f"{page_number + 1} of {filename}: {e}"
                    )

            pdf.close()

            if not documents:

                print(
                    f"Warning: No readable text found in {filename}"
                )

            return documents

        except Exception as e:

            print(
                f"Warning: Could not load PDF "
                f"{filename}: {e}"
            )

            return []

    # =========================================================
    # TXT
    # =========================================================

    if file_path.lower().endswith(".txt"):

        try:

            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

            documents = loader.load()

            for document in documents:

                document.metadata["source"] = filename

            return documents

        except UnicodeDecodeError:

            try:

                loader = TextLoader(
                    file_path,
                    encoding="latin-1"
                )

                documents = loader.load()

                for document in documents:

                    document.metadata["source"] = filename

                return documents

            except Exception as e:

                print(
                    f"Warning: Could not load text file "
                    f"{filename}: {e}"
                )

                return []

        except Exception as e:

            print(
                f"Warning: Could not load text file "
                f"{filename}: {e}"
            )

            return []

    # =========================================================
    # UNSUPPORTED FILE
    # =========================================================

    print(
        f"Warning: Unsupported file type: {filename}"
    )

    return []