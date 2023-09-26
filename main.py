# fastAPI 실행 코드
# python -m uvicorn main:app --reload
# python -m uvicorn main:app --reload --host=0.0.0.0 --port=8000
import openai
import dotenv
import os

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

key = os.environ["OPENAI_API_KEY"]
openai.api_key = key

from fastapi import FastAPI

# Sqlite DataBase
from database import models
from database.sqlite import engine
models.Base.metadata.create_all(bind=engine)

from domain import topic_recommend,chatbot,modification

app = FastAPI()


app.include_router(topic_recommend.router)
app.include_router(chatbot.router)
app.include_router(modification.router)