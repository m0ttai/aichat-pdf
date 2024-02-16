import qdrant_client
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

client = qdrant_client.QdrantClient(path="./local_qdrant")
qdrant = Qdrant(
    client=client,
    collection_name="my_collection",
    embeddings=OpenAIEmbeddings()
)
query = "Amazon FSx for NetApp ONTAP とは何か"
docs = qdrant.similarity_search(query=query, k=2)
for i in docs:
    print({"content": i.page_content, "metadata": i.metadata})
