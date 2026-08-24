"""
ingest.py

Reads master_bank.md, splits it into individual entries, converts each
one into a vector using a local embedding model, and stores everything
in a local ChromaDB database so it can be searched later.

Safe to re-run any time master_bank.md changes -- existing entries get
updated in place, new entries get added, nothing gets duplicated.
"""

import sys

import chromadb
import ollama

MASTER_BANK_PATH = "master_bank.md"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "master_bank"
EMBEDDING_MODEL = "nomic-embed-text"


def parse_master_bank(file_path):
    """
    Split master_bank.md into a list of entries.

    Each entry in the file looks like:

        ## Title
        **Category:** ...
        **Source:** ...
        **Skills:** ...

        Body text.

    Returns a list of dictionaries, one per entry, each with the keys
    title, category, source, skills, and body.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Every entry starts with a line that begins "## ". Splitting on
    # that pattern gives us one chunk of text per entry. The very
    # first piece is the document's title and intro text, which we
    # don't need, so we skip it with [1:].
    raw_pieces = content.split("\n## ")

    entries = []
    for piece in raw_pieces[1:]:
        lines = piece.strip().split("\n")
        title = lines[0].strip()

        category = ""
        source = ""
        skills = ""
        body_lines = []

        for line in lines[1:]:
            if line.startswith("**Category:**"):
                category = line.replace("**Category:**", "").strip()
            elif line.startswith("**Source:**"):
                source = line.replace("**Source:**", "").strip()
            elif line.startswith("**Skills:**"):
                skills = line.replace("**Skills:**", "").strip()
            elif line.strip() == "---":
                # A section divider in the file, not part of any
                # entry's actual content.
                continue
            else:
                body_lines.append(line)

        body = "\n".join(body_lines).strip()

        entries.append({
            "title": title,
            "category": category,
            "source": source,
            "skills": skills,
            "body": body,
        })

    return entries


def get_embedding(text):
    """
    Convert a piece of text into a vector using the local embedding
    model, via Ollama running on this machine.
    """
    try:
        result = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return result["embedding"]
    except Exception as e:
        print(
            f"\nCould not reach the local embedding model.\n"
            f"Two likely causes:\n"
            f"  1. Ollama isn't running right now -- open the Ollama app, "
            f"or run 'ollama serve' in a separate terminal.\n"
            f"  2. The '{EMBEDDING_MODEL}' model isn't pulled -- run "
            f"'ollama pull {EMBEDDING_MODEL}'.\n\n"
            f"Underlying error: {e}"
        )
        sys.exit(1)


def main():
    print(f"Reading {MASTER_BANK_PATH} ...")
    entries = parse_master_bank(MASTER_BANK_PATH)
    print(f"Found {len(entries)} entries.\n")

    # A quick sanity check before spending time embedding anything --
    # if any entry is missing a field, the file has a formatting
    # mistake worth fixing before continuing.
    incomplete = [
        e for e in entries
        if not e["category"] or not e["source"] or not e["skills"] or not e["body"]
    ]
    if incomplete:
        print("The following entries are missing a field (Category, Source, Skills, or body text):")
        for e in incomplete:
            print(f"  - {e['title']}")
        print("\nFix these in master_bank.md, then run this script again.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, entry in enumerate(entries):
        print(f"Embedding entry {i + 1}/{len(entries)}: {entry['title']}")

        # The embedding is generated from the title plus the body,
        # not the body alone -- the title often carries meaningful
        # signal (e.g. "Kubernetes" appearing in a title) that helps
        # retrieval later, even if the body phrases things differently.
        text_to_embed = f"{entry['title']}\n{entry['body']}"

        ids.append(f"entry_{i:03d}")
        embeddings.append(get_embedding(text_to_embed))
        documents.append(entry["body"])
        metadatas.append({
            "title": entry["title"],
            "category": entry["category"],
            "source": entry["source"],
            "skills": entry["skills"],
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"\nDone. {collection.count()} entries are now stored in the local database at {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()
    