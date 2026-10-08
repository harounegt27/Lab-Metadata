import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "invoices"

FILTER_FIELDS = [
    "invoice_number", "date", "bill_to", "ship_to",
    "item", "quantity", "amount", "discount", "shipping", "total"
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Invoice RAG",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 Invoice RAG Assistant")
st.caption("Search and chat with your SuperStore invoices powered by Qdrant + Groq")


# ── Load models (cached so they load only once) ───────────────────────────────
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-20b",
    )
    client = QdrantClient(url=QDRANT_URL)
    return vector_store, llm, client

vector_store, llm, qdrant_client = load_resources()


# ── Prompts ───────────────────────────────────────────────────────────────────
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

rewrite_chain = rewrite_prompt | llm | StrOutputParser()


def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


# ── Layout: sidebar + main ────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Metadata Filters")
    st.caption("Narrow search to a specific invoice field")

    filter_field = st.selectbox(
        "Filter field",
        options=["None"] + FILTER_FIELDS,
    )

    filter_value = ""
    if filter_field != "None":
        filter_value = st.text_input(f"Value for `{filter_field}`")

    st.divider()
    st.header("📊 Collection Stats")
    try:
        count = qdrant_client.count(collection_name=COLLECTION_NAME).count
        st.metric("Total invoices indexed", count)
    except:
        st.warning("Could not reach Qdrant")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Results table (last retrieved docs) ──────────────────────────────────────
if "last_docs" in st.session_state and st.session_state.last_docs:
    with st.expander("📄 Retrieved invoice chunks", expanded=False):
        for i, doc in enumerate(st.session_state.last_docs, 1):
            m = doc.metadata
            st.markdown(f"**Result {i}** — `{m.get('source_file', '?')}`")
            cols = st.columns(5)
            cols[0].metric("Invoice #", m.get("invoice_number", "—"))
            cols[1].metric("Date", m.get("date", "—"))
            cols[2].metric("Bill To", m.get("bill_to", "—"))
            cols[3].metric("Total", m.get("total", "—"))
            cols[4].metric("Item", m.get("item", "—"))
            st.caption(doc.page_content[:300] + "...")
            st.divider()


# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask something about your invoices..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Build filter if set
    metadata_filter = None
    if filter_field != "None" and filter_value.strip():
        metadata_filter = Filter(
            must=[
                FieldCondition(
                    key=f"metadata.{filter_field}",
                    match=MatchValue(value=filter_value.strip())
                )
            ]
        )

    # Retrieve + answer
    with st.chat_message("assistant"):
        with st.spinner("Searching invoices..."):
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

            # Get docs separately for the results table
            docs = retriever.invoke(question)
            st.session_state.last_docs = docs

            answer = rag_chain.invoke(question)

        st.markdown(answer)

        # Show active filter badge
        if metadata_filter:
            st.caption(f"🔎 Filtered by `{filter_field}` = `{filter_value}`")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()