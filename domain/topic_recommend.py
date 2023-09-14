from fastapi import APIRouter
from typing import Union
from pydantic import BaseModel
import json
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import openai
import dotenv
import os
from resource.model.openai_llms import OpenAISubjectRecommander

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

key = os.environ["OPENAI_API_KEY"]
openai.api_key = key

router = APIRouter(
    prefix="/topic_recommand",
)
###############################
# 1. 카테고리에 대한 주제 생성
###############################
subject_recommander = OpenAISubjectRecommander()

class Subject(BaseModel):
    category: Union[str, None] = None

@router.post("/make_subject")
def make_subject(item: Subject):
    print(item)
    category = item.dict()['category']
    subjects =subject_recommander.get_answer(category)
    
    return JSONResponse(content=jsonable_encoder(subjects))
    # return subjects

### sample
class Server(BaseModel):
    server: Union[str, None] = None

@router.post("/for_server")
def for_server(item:Server):
    print(item)
    a="hello"
    b = {"key": "result"}
    # print(type(b))
    return a

###############################
# 2. 오늘의 주제 or 직접 작성 시
###############################
class Category(BaseModel):
    subject: Union[str, None] = None

@router.post("/make_category")
def make_category(item: Category):
    subject = item.dict()['subject']
    print(subject)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "category=[일상, 사회, 과학,스포츠, 문화/예술,환경]\nUser가 입력한 값을 category 리스트 중 1개로 분류해주세요.\n문자열로 출력해야 합니다."
            },
            {
                "role": "user",
                "content": subject
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    category = response['choices'][0]['message']['content'].replace('\'','')
    return {'category': category}
