import os
import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from metadata_extractor import extract_invoice_metadata

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "invoices"
FOLDER_PATH = Path("invoices")
BATCH_SIZE = 50  # insert in batches to avoid memory issues


# 1. Embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect or prepare collection

client = QdrantClient(url=QDRANT_URL)

if client.collection_exists(collection_name=COLLECTION_NAME):
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    print(f"Connected to existing collection: '{COLLECTION_NAME}'")
else:
    vector_store = None
    print(f"Collection '{COLLECTION_NAME}' will be created.")


# 3. Helpers

def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def hash_exists_in_qdrant(file_hash):
    if vector_store is None:
        return False
    points, _ = vector_store.client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={
            "must": [{"key": "metadata.file_hash", "match": {"value": file_hash}}]
        },
        limit=1,
    )
    return len(points) > 0


# 4. Load PDFs

all_documents = []

for root, _, files in os.walk(FOLDER_PATH):
    print("Processing folder:", root)
    for file in files:
        if not file.lower().endswith(".pdf"):
            continue

        full_path = os.path.join(root, file)
        file_hash = calculate_file_hash(full_path)

        if hash_exists_in_qdrant(file_hash):
            print(f"  Skipping already indexed: {file}")
            continue

        loader = PyPDFLoader(str(full_path))
        documents = loader.load()

        full_text = "\n".join(doc.page_content for doc in documents)
        invoice_meta = extract_invoice_metadata(full_text)

        print(f"  Loaded: {file} → invoice #{invoice_meta['invoice_number']} | "
              f"{invoice_meta['bill_to']} | total: {invoice_meta['total']}")

        for document in documents:
            document.metadata.update({
                "file_hash":      file_hash,
                "source_file":    str(full_path),
                "invoice_number": invoice_meta["invoice_number"],
                "date":           invoice_meta["date"],
                "bill_to":        invoice_meta["bill_to"],
                "ship_to":        invoice_meta["ship_to"],
                "item":           invoice_meta["item"],
                "quantity":       invoice_meta["quantity"],
                "amount":         invoice_meta["amount"],
                "discount":       invoice_meta["discount"],
                "shipping":       invoice_meta["shipping"],
                "total":          invoice_meta["total"],
            })

        all_documents.extend(documents)

print(f"\nLoaded {len(all_documents)} pages total.")


# 5. Split

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)
chunks = splitter.split_documents(all_documents)
print(f"Created {len(chunks)} chunks.")


# 6. Insert in batches

def insert_batches(chunks, vector_store=None):
    global COLLECTION_NAME
    total = len(chunks)

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        end = min(i + BATCH_SIZE, total)
        print(f"  Inserting batch {i+1}–{end} / {total} ...", end=" ")

        try:
            if vector_store is None:
                # First batch — create the collection
                vector_store = QdrantVectorStore.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    url=QDRANT_URL,
                    collection_name=COLLECTION_NAME,
                )
            else:
                vector_store.add_documents(batch)

            # Verify batch was inserted
            count = client.count(collection_name=COLLECTION_NAME).count
            print(f"✓  total points in Qdrant: {count}")

        except Exception as e:
            print(f"\n  ✗ ERROR on batch {i+1}–{end}: {e}")
            continue

    return vector_store


print("\nInserting into Qdrant...")
vector_store = insert_batches(chunks, vector_store)

# 7. Final verification
final_count = client.count(collection_name=COLLECTION_NAME).count
print(f"\n✓ Done — {final_count} points in collection '{COLLECTION_NAME}'.")