from langchain.chat_models import ChatOpenAI
from langchain.schema import (
	SystemMessage,  # システムメッセージ
	HumanMessage,  # 人間の質問
	AIMessage  # ChatGPTの返答
)

llm = ChatOpenAI()  # ChatGPT APIを呼んでくれる機能
# message = "Hi, ChatGPT!"  # あなたの質問をここに書く
message = "こんにちはやで！"  # あなたの質問をここに書く

messages = [
	# SystemMessage(content="You are a helpful assistant."),
	SystemMessage(content="必ず関西弁で返答してください"),
	HumanMessage(content=message)
]
response = llm(messages)
print(response)

# content='Hello! How can I assist you today?' additional_kwargs={} example=False
