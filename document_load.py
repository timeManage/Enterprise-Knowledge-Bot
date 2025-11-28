import xml.dom.minidom

import chromadb
import dotenv
import langchain_classic.chains.retrieval_qa.base
import langchain_community.embeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import os

dotenv.load_dotenv()
api_key = os.environ.get('API_KEY')
proxy_url = "http://172.16.137.158:1081"
os.environ["HTTP_PROXY"] = proxy_url
os.environ["HTTPS_PROXY"] = proxy_url

def split_docuemnt(file_path:str='/Users/shentao/Downloads/2022张宇数学命题人终极预测8套卷-数学一-解析册（过关版）.pdf'):
    loader = PyPDFLoader(file_path)

    pages = loader.load()

    # print(f"一共加载了 {len(pages)} 页")
    # print("第一页的内容是：")
    # print(pages[0].page_content)

    full_content = "\n".join([page.page_content for page in pages])

    # 3. 重组：把这个巨大的字符串包装成一个新的 Document 对象
    # 这里我们手动造了一个 Document
    merged_document = Document(page_content=full_content, metadata={"source": "merged_pdf"})

    # 4. 切割：现在切割器面对的是连贯的整体
    # 它会根据语义（换行、句号）来切，完全无视之前的物理分页
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents([merged_document])

    # 5. 验证跨页重叠
    print(f"合并后总长度: {len(full_content)} 字符")
    print(f"切割后段数: {len(splits)}")
    return splits



def reset_knowledge():
    global vectorstore
    # 假设你要删除的文件路径或名称是这个
    target_file_source = "merged_pdf"

    # 1. 【查找】使用 get 方法配合 where 过滤条件，找到所有属于该文件的 ID
    # Chroma 支持根据 metadata 字段进行过滤
    record = vectorstore.get(
        where={"source": target_file_source}
    )
    test_result = vectorstore.get(limit=1)
    if test_result['metadatas']:
        print("当前 Metadata 结构示例:", test_result['metadatas'][0])
    ids_to_delete = record['ids']

    print(f"找到属于 {target_file_source} 的切片数量: {len(ids_to_delete)}")

    # 2. 【删除】如果有找到 ID，则进行删除
    if len(ids_to_delete) > 0:
        vectorstore.delete(ids=ids_to_delete)
        print(f"成功删除文档: {target_file_source}")
    else:
        print("未找到该文档，请检查 metadata 中的 source 字段是否匹配。")
    # vectorstore.delete_collection()
    # vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, client_settings=chromadb.Settings(allow_reset=True))

embeddings = langchain_community.embeddings.HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

llm = ChatOpenAI(
    model="gemini-2.5-flash",  # 模型名称 (DeepSeek官网叫这个)
    api_key=api_key,  # 填你之前申请的 DeepSeek Key
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # 【关键】把地址指向 DeepSeek，而不是默认的 OpenAI
    temperature=0,  # 0 表示回答严谨，不随机发散
    openai_proxy="http://172.16.137.158:1081",
)
# 构建 RAG 链
qa_chain = langchain_classic.chains.retrieval_qa.base.RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # k=3 意思是只找最相似的3段
    return_source_documents=True  # 让我们看看它参考了哪几段
)
print("RAG 链已更新，绑定了最新的知识库。")
print("向量数据库构建完成！")