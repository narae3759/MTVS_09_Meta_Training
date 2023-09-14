# fastAPI 실행 코드
# python -m uvicorn main:app --reload
# python -m uvicorn main:app --reload --host=0.0.0.0 --port=8000
import openai
import dotenv
import os
import json

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

key = os.environ["OPENAI_API_KEY"]
openai.api_key = key

from domain import topic_recommend,chatbot,modification


from typing import Union
from fastapi import FastAPI

app = FastAPI()


app.include_router(topic_recommend.router)
app.include_router(chatbot.router)
app.include_router(modification.router)