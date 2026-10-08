import os
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "invoices"

# All filterable metadata fields from the invoice mapping
FILTER_FIELDS = [
    "invoice_number",
    "date",
    "bill_to",
    "ship_to",
    "item",
    "quantity",
    "amount",
    "discount",
    "shipping",
    "total",
]


# 1. LLM

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)


# 2. Prompts

prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant specialized in invoice analysis.
    Always answer the question given the context.
    If you don't have an answer, just say: I don't know.

    Question:
    {question}

    Context:
    {context}
""")

rewrite_prompt = ChatPromptTemplate.from_template("""
    Rewrite the user's question into a concise search query.
    Do not answer the question.
    Keep the important keywords.

    Question:
    {question}

    Search query:
""")


# 3. Embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Connect to existing collection

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)


# 5. Optional metadata filter

def build_filter():
    print(f"\n  Available filters: {', '.join(FILTER_FIELDS)}")
    field = input("  Filter field (press Enter to skip): ").strip()
    if not field or field not in FILTER_FIELDS:
        return None
    value = input(f"  Value for '{field}': ").strip()
    if not value:
        return None
    return Filter(
        must=[
            FieldCondition(
                key=f"metadata.{field}",
                match=MatchValue(value=value)
            )
        ]
    )


# 6. Helpers

def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)

rewrite_chain = rewrite_prompt | llm | StrOutputParser()


# 7. Chat loop

print("Invoice RAG — type /quit to exit.\n")

while True:
    question = input("Ask something: ").strip()
    if question.lower() in ["/quit", "/bye"]:
        print("Exiting.")
        break

    metadata_filter = build_filter()

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5,
            **({"filter": metadata_filter} if metadata_filter else {}),
        }
    )

    rag_chain = (
        {
            "context": rewrite_chain | retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    print(f"\nAI: {answer}\n")