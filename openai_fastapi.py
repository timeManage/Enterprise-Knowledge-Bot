import fastapi
import uvicorn

from gemini import gemini_2_5_flash,QuestionModel

app = fastapi.FastAPI(
    title="openai接口",
    version="1.0.0",
    description="***"
)


@app.post('/v1/chat/')
async def chat(questionModel: QuestionModel):
    """
    Ask a question to the model.
    """
    response = gemini_2_5_flash(questionModel)
    content= response.choices[0].message.content
    questionModel.history.append({"role": "system", "content": content})
    return content


if __name__ == '__main__':
    uvicorn.run('openai_fastapi:app', host="0.0.0.0", port=8000, log_level="info", reload=True)
