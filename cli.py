import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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


def main():

	if len(sys.argv) < 2:
		print("Usage: python cli.py \"your question\"")
		return

	question = sys.argv[1]
	chain = qa_bot()
	print(chain.invoke(question))


if __name__ == "__main__":

	main()


