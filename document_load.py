import xml.dom.minidom

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
embeddings = langchain_community.embeddings.HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# 2. 准备向量库
# 如果 ./chroma_db 目录存在，直接读取；不存在则重新构建
persist_dir = "./chroma_db"
if os.path.exists(persist_dir):
    print("发现现有向量库，正在加载...")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    print("未发现向量库，正在从 test.pdf 构建...")
    # 这里放你刚才跑通的加载、切割、构建代码
    loader = PyPDFLoader("./test.pdf")
    pages = loader.load()
    # ... (加上你那个合并页面的逻辑) ...
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(pages)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_dir)

print("向量数据库构建完成！")
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
    return_source_documents=True # 让我们看看它参考了哪几段
)
# 5. 测试
# question = "在这个文档中，怎么判断病例是属于床日的？" # 换成针对你 PDF 的问题
# print(f"用户提问: {question}")
# result = qa_chain.invoke({"query": question})
#
# print("--- AI 回答 ---")
# print(result["result"])
# print("\n--- 参考片段 (证据) ---")
# for doc in result["source_documents"]:
#     print(f"[内容]: {doc.page_content[:50]}...") # 只打印前50个字