from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chainlit as cl
from langchain_community.cache import SQLiteCache
import time
import asyncio


DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """
You are an expert in homeopathy. Based on the provided context, answer the user's question concisely and only provide the information requested.

Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) > max_tokens:
        return " ".join(tokens[-max_tokens:])
    return text

# Use this when preparing context before feeding it to the model


def build_rag_chain(llm, prompt, db):
    retriever = db.as_retriever(search_kwargs={'k': 1})

    def join_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | join_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=200,
        temperature=0.6,
        top_p=0.9
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    return build_rag_chain(llm, qa_prompt, db)

def remove_redundant_phrases(text):
    # Basic approach to remove near-duplicate sentences
    sentences = text.split('. ')
    seen_sentences = set()
    filtered_sentences = []
    for sentence in sentences:
        if sentence not in seen_sentences:
            filtered_sentences.append(sentence)
            seen_sentences.add(sentence)
    return '. '.join(filtered_sentences)

def final_result(query):
    chain = qa_bot()
    answer = chain.invoke(query)
    return remove_redundant_phrases(answer)


@cl.on_chat_start
async def start():
    chain = qa_bot()
    intro = (
        "Welcome to Medical RAG Assistant.\n\n"
        "Ask a clinical question (include patient context).\n"
        "Answers include citations from your PDF corpus.\n"
        "This is not a substitute for professional medical advice."
    )
    await cl.Message(content=intro).send()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    question = message.content
    t0 = time.time()

    # Log incoming question to server console for debugging
    print(f"[Chainlit] Question: {question[:200]}")

    # Guard: if chain is missing, rebuild once
    if chain is None:
        print("[Chainlit] Chain missing in session. Rebuilding...")
        chain = qa_bot()
        cl.user_session.set("chain", chain)

    # Run with a timeout so UI doesn't hang silently
    try:
        # Some LangChain components block under ainvoke; use make_async wrapper
        answer = await asyncio.wait_for(cl.make_async(chain.invoke)(question), timeout=180)
    except asyncio.TimeoutError:
        await cl.Message(content="The model is taking too long. Please try again or simplify the query.").send()
        print("[Chainlit] Timeout while generating answer")
        return
    except Exception as e:
        await cl.Message(content=f"Error generating answer: {e}").send()
        print(f"[Chainlit] Error: {e}")
        return
    dt_ms = int((time.time() - t0) * 1000)

    # Deduplicate whitespace
    answer = " ".join(dict.fromkeys(answer.split()))

    # Fetch top sources for display
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={'k': 3})
        docs = retriever.get_relevant_documents(question)
        citations = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get('source') or d.metadata.get('file_path') or 'source.pdf'
            page = d.metadata.get('page')
            page_str = f": p.{page}" if page is not None else ""
            snippet = (d.page_content or "").strip().replace('\n', ' ')
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."
            citations.append(f"[{i}] {src}{page_str} — {snippet}")
        sources_text = "\n".join(citations) if citations else "No sources found."
        elements = [cl.Text(name="Sources", content=sources_text)]
    except Exception as e:
        print(f"[Chainlit] Citation retrieval error: {e}")
        elements = []

    footer = "\n\nNote: This information is for educational purposes and not medical advice."
    await cl.Message(content=f"{answer}\n\n_Response time: {dt_ms} ms_{footer}", elements=elements).send()


@cl.action_callback("example_1")
async def on_example_1(action: cl.Action):
    await main(cl.Message(content=(action.payload or {}).get("text", "")))


@cl.action_callback("example_2")
async def on_example_2(action: cl.Action):
    await main(cl.Message(content=(action.payload or {}).get("text", "")))


@cl.action_callback("example_3")
async def on_example_3(action: cl.Action):
    await main(cl.Message(content=(action.payload or {}).get("text", "")))
