# rag_app_on_vm.py

from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

# Databricks CLI로 설정된 인증 정보를 자동으로 사용합니다.
# 1. Vector Search에 연결하여 검색기(Retriever) 만들기
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
vector_store = DatabricksVectorSearch(
    endpoint_name="YOUR_VS_ENDPOINT_NAME",
    index_name="YOUR_VS_INDEX_NAME",
    embedding=embedding_model
)
retriever = vector_store.as_retriever()

# 2. Databricks의 LLM 모델에 연결
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=200)

# 3. RAG 체인 구성 (이 부분은 노트북 코드와 완전히 동일)
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever
)

# 4. VSCode 터미널에서 바로 실행하고 결과 확인
if __name__ == "__main__":
    question = "내 문서에 대해 알려줘"
    response = qa_chain.invoke({"query": question})
    print("--- 답변 ---")
    print(response["result"])