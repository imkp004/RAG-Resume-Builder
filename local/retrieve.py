"""
retrieve.py

Takes a job description and finds the most relevant entries from the
master experience bank, using the local database built in ingest.py.

Can be run directly to test retrieval quality on its own, or imported
by other scripts -- generate.py in the next step will do exactly this.
"""

import sys

import chromadb
import ollama

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "master_bank"
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_RESULT_COUNT = 10


def embed_text(text):
    """
    Convert a piece of text into a vector, using the same local
    model that was used when the database was first built. Using
    the same model both times is required -- vectors from two
    different embedding models are not comparable to each other.
    """
    try:
        result = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return result["embedding"]
    except Exception as e:
        print(
            f"\nCould not reach the local embedding model.\n"
            f"Check that the Ollama application is running, and that "
            f"the '{EMBEDDING_MODEL}' model has been pulled.\n\n"
            f"Underlying error: {e}"
        )
        sys.exit(1)


def retrieve_relevant_chunks(job_description_text, result_count=DEFAULT_RESULT_COUNT):
    """
    Given a job description, return the most relevant entries from
    the master experience bank.

    Returns a list of dictionaries, each containing the entry's
    title, category, source, skills, the actual text, and a distance
    number (lower means a stronger match).
    """
    query_vector = embed_text(job_description_text)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=result_count,
    )

    matches = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        matches.append({
            "title": metadata["title"],
            "category": metadata["category"],
            "source": metadata["source"],
            "skills": metadata["skills"],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
        })

    return matches


def print_matches(matches):
    """
    Print retrieval results in a readable way, so you can manually
    check whether retrieval quality looks right before moving on to
    having a model write anything from these results.
    """
    print(f"\nTop {len(matches)} matches (closest match first -- a lower")
    print("distance number means a stronger match):\n")

    for rank, match in enumerate(matches, start=1):
        print(f"{rank}. {match['title']}  (distance: {match['distance']:.2f})")
        print(f"   Category: {match['category']}")
        print(f"   Source: {match['source']}")
        print(f"   Skills: {match['skills']}")
        print(f"   {match['text'][:150]}{'...' if len(match['text']) > 150 else ''}")
        print()


def main():
    print("Paste a job description below.")
    print("When finished, press Enter, then press Control-D to submit.\n")

    job_description_text = sys.stdin.read().strip()

    if not job_description_text:
        print("No job description was entered. Exiting.")
        sys.exit(1)

    print("\nSearching the experience bank ...")
    matches = retrieve_relevant_chunks(job_description_text)
    print_matches(matches)


if __name__ == "__main__":
    main()
    