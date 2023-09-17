import openai
import dotenv
import os
import json
from typing import Union
from pydantic import BaseModel

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

key = os.environ["OPENAI_API_KEY"]
openai.api_key = key

from fastapi import APIRouter


router = APIRouter(
    prefix="/modification",
)

class Title(BaseModel):
    content: Union[str, None] = None

@router.post("/make_title")
def make_title(item: Title):
    content = item.dict()['content']
    system_text='''
    1.사용자의 글을 읽고 적당한 제목을 1개만 추천해줘
    2.사용자의 입력값은 <제목^서론^본론^결론> 순서로 되어있어
    3.출력형식은 Json형식으로 title이라는 키값을 가져야해
    '''

    pre_input1 = '''
    실시간 검색어 서비스 중단^실시간 검색어 서비스 중단은 좋은 방안이 아니였다.^실시간 검색어라는 것은 사람들이 무엇에 관심 있는지 알 수있는 지표기 때문이고 실시간 검색어를 통해 현재 가장 논란이 되고있는 뉴스나 사건들을 접하여 새로운 정보를 알수가있다.^실시간 검색어 서비스를 중단하지 않고 다른 방안을 생각하는 것이 더 나은 방안이라고 생각한다.
    '''

    pre_output1 = '''
    {"title": "실시간 검색어 서비스 중단 문제와 대안에 대한 고민"}
    '''


    pre_input2 = '''
    촉법소년 체벌에 대한 논쟁^현대 사회에서 촉법소년 문제는 많은 이들의 이목을 끌고 있는 논쟁적인 주제 중 하나로, 이에 대한 다양한 의견과 입장이 존재합니다. 촉법소년에 대한 처벌 방안은 그 특수성으로 인해 논의가 분분하며, 이는 사회의 안전과 촉법소년의 교육적 필요 사이에서의 균형을 찾는 것이 중요합니다.^촉법소년에 대한 체벌은 두 가지 주요 입장으로 나뉩니다. 먼저, 일부는 엄격한 처벌을 주장합니다. 그들은 촉법소년이 범행을 저질렀을 때, 이를 엄격한 처벌로 직면하게 함으로써 사회에 경고를 보내고 범죄를 줄일 수 있다고 주장합니다. 또한, 이들은 촉법소년이 자신의 행동에 대한 책임을 느끼게 하기 위한 필요성을 강조합니다.다른 한편으로는, 촉법소년에 대한 교육적인 접근을 주장하는 사람들도 있습니다. 이들은 촉법소년이 범행의 결과와 그에 따르는 처벌의 의미를 충분히 이해하고 교정을 통해 다시 사회에 복귀할 수 있도록 지원해야 한다고 주장합니다. 또한, 촉법소년이 더 나은 선택을 할 수 있도록 도움을 주는 교육적 프로그램의 중요성을 강조합니다.^촉법소년에 대한 체벌은 사회적 안전과 촉법소년의 교육적인 필요 사이에서 균형을 찾아야 하는 중요한 과제입니다. 엄격한 처벌과 교육적 접근은 각각의 장단점을 가지고 있으며, 이를 조화롭게 결합하는 방안을 모색해야 합니다. 최종적으로, 촉법소년에게는 그들의 행동에 대한 책임을 느끼게 하고, 더 나은 선택을 할 수 있도록 지원하는 시스템이 구축되어야 합니다.
    '''

    pre_output2 = '''
    {"title": "촉법소년 체벌에 대한 논쟁과 균형을 찾기 위한 대안 고민"}
    '''

    messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": pre_input1},
            {"role": "assistant", "content": pre_output1},
            {"role": "user", "content": pre_input2},
            {"role": "assistant", "content": pre_output2},
            {"role": "user", "content": content},
        ]

    chat_completion = openai.ChatCompletion.create( ## gpt 오브젝트 생성후 메세지 전달
        model="gpt-4", 
        messages=messages,
        temperature=1,
        max_tokens=1000
        )

    make_t = chat_completion.choices[0].message.content  ## gpt결과값 출력
    output = json.loads(make_t)
    return output


class Check(BaseModel):
    content: Union[str, None] = None

@router.post("/check_content")
def check_content(item: Check):
    content = item.dict()['content']
    system_text='''
    1.사용자의 입력값은 <주제^서론^본론^결론> 순서로 되어있어
    2.입력값에서 아래의 규칙에 해당되는 부분을 3가지 찾아줘.
    1) 반말과 존댓말이 모두 사용되었다
    2) 문장의 연결 접속사가 어색하다
    3) 내용의 흐름이 잘 이어지지 않는다
    3. 찾은 문장을 그대로 mistakes 변수에 리스트로 표현해줘.
    4. 이유와 대안을 존댓말로 청소년의 시선에 맞게 reasons, advices에 리스트로 표현해줘
    5. 대안은 주제에서 벗어나지 않아야 하고, 만약 찾은 문장이 본론과 무관하다면 "(삭제)"라고 표현해줘. 
    6. json 형식으로 출력해줘
    '''

    pre_input1 = '''
    가장 여행 가고 싶은 나라^여행이고 싶는 나라는 이태리 입니다. 이태리에 가면 피자와 파스타가 너무 맛있죠. 그리고 로마에 있는 고대 유적지들은 정말 믿기 힘들 정도로 아름다워요.^근데 나는 지금 배가고파서 글을 쓰기가 싫어 일본에 맛있는 밥집이 있는데 맛이 어땠더라?
    어쩄든 나는 도쿄에서 밥을 머코시퍼^여행지를 선택하는 건 어렵죠. 하지만 나는 어디든 간다면 즐겁게 여행할 거예요!
    '''

    pre_output1 = '''
    {
        "mistakes": ["근데 나는 지금 배가고파서 글을 쓰기가 싫어", "일본에 맛있는 밥집이 있는데 맛이 어땠더라?", "어쩄든 나는 도쿄에서 밥을 머코시퍼"],
        "reasons": ["느닷없이 '배가고파서 글쓰기를 싫어한다는' 내용은 본론과 무관하며, 반말을 사용했습니다",
                    "본문은 이태리 여행에 관한 내용이지만, 글 중간에 어떤 일본 밥집의 맛에 대해 언급하며 흐름이 이어지지 않았습니다",
                    "글은 이태리 여행에 의견에 관한 내용이며, '도쿄에서 밥을 먹고싶다'는 문장은 본론에서 벗어나고, 발표의 경중을 떨어뜨리는 반말이 사용되었습니다"],
        "advices": ["'그러나 현재 저는 배가 고파 글쓰기가 조금 힘들어요.'라는 문장을 쓰되 존댓말을 사용해보세요. 하지만 이 부분은 본론과 관련이 없어서 삭제하는 것이 더 좋을 것 같네요",
                    "글의 흐름을 위해서 이 부분은 삭제하는 것을 권장합니다",
                    "'도쿄에서 밥을 먹고 싶다'는 문장은 본론과 관련이 없으므로 삭제하는 것이 좋아 보입니다"]
    }
    '''

    pre_input2 = '''
    대체육은 고기인가 아닌가^대체육은 고기가 아니다.^대체육이라는 이름의 의미까지 살펴보아야한다. 대체육이란 대체+육이라는 단어가 합쳐진 것으로 물고기를 대체 한다는 의미이기 떄문에 대체육은 고기가 아니다. 대표적인 대체육으로는 콩고기가있다. 콩고기는 콩으로 만들어져있기때문에 나는 자주먹는다. 영양성분자체가 고기와 완전히 다르다.^따라서 대체육은 고기가 아닐다.
    '''

    pre_output2 = '''
    {
        "mistakes": ["아 배고프다.", "콩고기는 콩으로 만들어져있기때문에 나는 자주먹는다."],
        "reasons": ["'아 배고프다.'는 논문의 흐름과 관련이 없는 문장입니다.", "'나는 자주먹는다.'는 주어가 나타내는 1인칭 표현이 갑작스레 등장해 본문의 일관성을 해칩니다."],
        "advices": ["(삭제)", "'나는 자주먹는다.' 부분을 '콩으로 만든 '대체육은 건강에 좋고, 맛있기 때문에 많은 사람들이 선호합니다.' 등으로 바꾸면 어떨까요?"]
    }
    '''

    messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": pre_input1},
            {"role": "assistant", "content": pre_output1},
            {"role": "user", "content": pre_input2},
            {"role": "assistant", "content": pre_output2},
            {"role": "user", "content": content},
        ]

    chat_completion = openai.ChatCompletion.create( ## gpt 오브젝트 생성후 메세지 전달
        model="gpt-4",
        messages=messages,
        temperature=1,
        max_tokens=1000
        )

    check_c = chat_completion.choices[0].message.content  ## gpt결과값 출력
    output = json.loads(check_c)
    return output