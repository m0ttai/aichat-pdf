from glob import glob
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_community.callbacks import get_openai_callback

from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import json

QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection_2"


def init_page():
	st.set_page_config(
		page_title="FSxN相談屋さんへようこそ！",
		page_icon="🤗"
	)


def select_model():
	# 300: 本文以外の指示のトークン数 (以下同じ)
	st.session_state.model_name = "gpt-3.5-turbo"
	st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
	return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)


# def get_pdf_text():
# 	uploaded_file = st.file_uploader(
# 		label='Upload your PDF here😇',
# 		type='pdf'
# 	)
# 	if uploaded_file:
# 		pdf_reader = PdfReader(uploaded_file)
# 		text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
# 		text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
# 			model_name="text-embedding-ada-002",
# 			# 適切な chunk size は質問対象のPDFによって変わるため調整が必要
# 			# 大きくしすぎると質問回答時に色々な箇所の情報を参照することができない
# 			# 逆に小さすぎると一つのchunkに十分なサイズの文脈が入らない
# 			chunk_size=500,
# 			chunk_overlap=0,
# 		)
# 		return text_splitter.split_text(text)
# 	else:
# 		return None


def load_qdrant():
	client = QdrantClient(path=QDRANT_PATH)

	# すべてのコレクション名を取得
	collections = client.get_collections().collections
	collection_names = [collection.name for collection in collections]

	# コレクションが存在しなければ作成
	if COLLECTION_NAME not in collection_names:
		# コレクションが存在しない場合、新しく作成します
		client.create_collection(
			collection_name=COLLECTION_NAME,
			vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
		)
		print('collection created')

	return Qdrant(
		client=client,
		collection_name=COLLECTION_NAME,
		embeddings=OpenAIEmbeddings()
	)


def build_vector_store(pdf_text):
	qdrant = load_qdrant()
	qdrant.add_texts(pdf_text)


def build_qa_model(llm):
	qdrant = load_qdrant()
	retriever = qdrant.as_retriever(
		# "mmr",  "similarity_score_threshold" などもある
		search_type="similarity",
		# 文書を何個取得するか (default: 4)
		search_kwargs={"k":8}
	)

	return RetrievalQA.from_chain_type(
		llm=llm,
		chain_type="stuff",
		retriever=retriever
	)


# def page_pdf_upload_and_build_vector_db():
# 	st.title("PDF Upload")
# 	container = st.container()
# 	with container:
# 		pdf_text = get_pdf_text()
# 		if pdf_text:
# 			with st.spinner("Loading PDF ..."):
# 				build_vector_store(pdf_text)


def ask(qa, query):
	with get_openai_callback() as cb:
		# query / result / source_documents
		answer = qa(query)

	return answer


def page_ask_my_pdf():
	st.title("Welcome to FSxN相談屋さん")

	llm = select_model()
	container = st.container()
	response_container = st.container()

	with container:
		query = st.text_input("†††相談内容を入力してください††† ", key="input")
		if not query:
			answer = None
		else:
			qa = build_qa_model(llm)
			if qa:
				with st.spinner("Please wait ..."):
					answer = ask(qa, query)
			else:
				answer = None

		if answer:
			answer_json = json.loads(json.dumps(answer))
			with response_container:
				st.markdown("# 相談屋さんからの回答はこちら")
				st.write(answer_json["result"])


def main():
	init_page()

	# selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask My PDF(s)"])
	# if selection == "PDF Upload":
	# 	page_pdf_upload_and_build_vector_db()
	# elif selection == "Ask My PDF(s)":
	# 	page_ask_my_pdf()
	page_ask_my_pdf()


if __name__ == '__main__':
	main()
