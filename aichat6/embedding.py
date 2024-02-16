# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

emb_model = OpenAIEmbeddings()
text = "こんにちは世界！"
result = emb_model.embed_documents([text])  # listで入れる必要があるので注意
print(f'Embeddingの次元数: {len(result[0])}')
print(result[0][:5])
