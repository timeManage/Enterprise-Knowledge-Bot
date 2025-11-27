import os
import shutil
import tempfile

import fastapi
import pydantic
from starlette.staticfiles import StaticFiles
import uvicorn

from document_load import qa_chain, split_docuemnt, vectorstore
from gemini import gemini_2_5_flash, QuestionModel

app = fastapi.FastAPI(
    title="openai接口",
    version="1.0.0",
    description="***"
)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(pydantic.BaseModel):
    userId: str
    sessionId: str
    question: str


questionModel = QuestionModel()


@app.post('/v1/chat/')
async def chat(chat_request: ChatRequest):
    """
    Ask a question to the model.
    """
    questionModel.history.append({"role": "user", "content": chat_request.question})
    # response = gemini_2_5_flash(questionModel)
    # content = response.choices[0].message.content
    content = qa_chain.invoke({"query": chat_request.question})
    print(content)
    result = content['result']
    questionModel.history.append({"role": "system", "content": result})
    return {'data': result, 'code': 200, 'msg': 'SUCCESS'}


class UpdateRequest(pydantic.BaseModel):
    file_path: str  # Java 告诉 Python 文件存在哪里了


@app.post("/update_knowledge")
async def update_knowledge(file: fastapi.UploadFile= fastapi.File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        # 2. 将上传的文件内容写入临时文件
        # noinspection PyTypeChecker
        shutil.copyfileobj(file.file, tmp_file)
        tmp_file_path = tmp_file.name # 获取临时文件路径
    try:
        print(f"--- 开始处理新文件: {tmp_file_path} ---")

        splits = split_docuemnt(tmp_file_path)
        # 注意：add_documents 会把新知识追加进去，而不是覆盖旧的
        vectorstore.add_documents(splits)

        # 4. 这一步很重要：持久化！否则重启就没了
        # Chroma 默认会在 add_documents 时自动持久化，但保险起见可以检查一下配置

        print(f"--- 新文件处理完毕，新增 {len(splits)} 个片段 ---")
        return {"code": 200, "msg": "知识库更新成功"}

    except Exception as e:
        print(f"Error: {e}")
        return {"code": 500, "msg": str(e)}


if __name__ == '__main__':
    uvicorn.run('openai_fastapi:app', host="0.0.0.0", port=8000, log_level="info", reload=True)
