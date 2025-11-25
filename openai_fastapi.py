import fastapi
import pydantic
from starlette.staticfiles import StaticFiles
import uvicorn

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
    response = gemini_2_5_flash(questionModel)
    content = response.choices[0].message.content
    questionModel.history.append({"role": "system", "content": content})
    return {'data': content, 'code': 200, 'msg': 'SUCCESS'}


if __name__ == '__main__':
    uvicorn.run('openai_fastapi:app', host="0.0.0.0", port=8000, log_level="info", reload=True)
