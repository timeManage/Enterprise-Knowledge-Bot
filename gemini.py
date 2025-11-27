import dotenv
import httpx
import pydantic
from openai import OpenAI
import openai
import os
dotenv.load_dotenv()
api_key = os.environ.get('API_KEY')

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    http_client=openai.DefaultHttpxClient(
        proxies="http://172.16.137.158:1081",
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    ),
)


class QuestionModel(pydantic.BaseModel):
    history: list = []
    question: str = ""


def gemini_2_5_flash(questionModel: QuestionModel):
    questionModel.history.append({"role": "user", "content": questionModel.question})
    if len(questionModel.history)>20:
        questionModel.history=[questionModel.history[0]+questionModel.history[-19:]]
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=questionModel.history
    )
    # print(response.choices[0].message.content)
    return response


# print(response.choices[0].message)

if __name__ == '__main__':
    gemini_2_5_flash("请将下面这段话翻译成英文：")
