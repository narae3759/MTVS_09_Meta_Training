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
    2.사용자의 글을 읽어보고 흐름이 어색한 부분이 있으면 말해줘
    3.어색한 부분이 있을때마다 번호를 매겨서 무조건 고쳐야 하는 부분만 말해줘
    4.어휘적으로 오류가 있으면 수정예시를 말해주고 문맥적으로 오류가있으면 오류에 대한 부분만 말을 해줘
    5.출력형식은 json으로 해줘, 어색한 부분이 3개가 있으면 어색한 문장은 mistake이라는 키값으로 리스트안에 3개, 조언에 대한 내용은 Advice라는 키 값으로 리스트안에 3개로 나타내줘
    6.출력은 최대 3개까지 나타내줘
    7. 문맥상 이상한 부분이 없으면 good이라는 키값으로 칭찬을해줘
    '''

    pre_input1 = '''
    가장 여행 가고 싶은 나라^여행이고 싶는 나라는 이태리 입니다. 이태리에 가면 피자와 파스타가 너무 맛있죠. 그리고 로마에 있는 고대 유적지들은 정말 믿기 힘들 정도로 아름다워요.^근데 나는 지금 배가고파서 글을 쓰기가 싫어 일본에 맛있는 밥집이 있는데 맛이 어땠더라?
    어쩄든 나는 도쿄에서 밥을 머코시퍼^여행지를 선택하는 건 어렵죠. 하지만 나는 어디든 간다면 즐겁게 여행할 거예요!
    '''

    pre_output1 = '''
    {
        "mistake":["근데 나는 지금 배가고파서 글을 쓰기가 싫어 일본에 맛있는 밥집이 있는데 맛이 어땠더라?","어쩄든 나는 도쿄에서 밥을 머코시퍼","여행지를 선택하는 건 어렵죠. 하지만 나는 어디든 간다면 즐겁게 여행할 거예요"]
        "Advice":["문장에 개인의 현재 상황이나 느낌을 이야기하는 것은 주제에서 벗어나는 경향이 있습니다. 아마도 이태리에 대한 이야기를 원하는 것이므로 일본에 대한 내용은 제외하고 이태리에 대한 설명에 집중해보는 것이 좋을 것 같습니다.","이 문장도 '여행 가고 싶은 나라'에 대한 주제를 벗어나 있어 보입니다. 본론에 해당하는 부분에서는 당신이 이태리를 선택한 이유나 그곳에서 느끼고자 하는 감정, 기대 등을 표현해 보는 것이 좋겠습니다. 예를 들어, '이태리의 맛있는 음식을 직접 맛보고 싶다'라는 식으로 수정해 볼 수 있겠습니다.","결론 부분이 주제인 '가장 여행 가고 싶은 나라'와 어울리지 않습니다. 이태리에 대한 기대나 마음가짐을 표현하거나, 이태리 방문을 통해 얻고자 하는 것이 무엇인지 기술하는 것이 좋을 것 같습니다."]
    }
    '''


    pre_input2 = '''
    대체육은 고기인가 아닌가^대체육은 고기가 아니다.^대체육이라는 이름의 의미까지 살펴보아야한다. 대체육이란 대체+육이라는 단어가 합쳐진 것으로 물고기를 대체 한다는 의미이기 떄문에 대체육은 고기가 아니다. 대표적인 대체육으로는 콩고기가있다. 콩고기는 콩으로 만들어져있기때문에 나는 자주먹는다. 영양성분자체가 고기와 완전히 다르다.^따라서 대체육은 고기가 아닐다.
    '''

    pre_output2 = '''
    {
        "mistake":["대표적인 대체육으로는 콩고기가있다. 콩고기는 콩으로 만들어져있기때문에 나는 자주먹는다.","콩고기는 콩으로 만들어져있기때문에 나는 자주먹는다.","따라서 대체육은 고기가 아닐다."]
        "Advice":["이 문장은 본론에 대해 설명하면서 갑자기 개인의 의견이나 행동을 언급하고 있어, 전체적인 글의 흐름을 방해합니다. 본론에서는 '대체육은 고기가 아니다'는 주장을 뒷받침하는 사실 위주로 서술하면 좋을 것 같습니다. 예를 들어, '콩고기는 고기와는 달리 식물성 단백질인 콩으로 만들어지며, 그 특성이 고기와는 다르다'라는 식으로 수정해보세요.","여기에서 '나는 자주 먹는다.' 라는 부분은 주제에 국한된 내용에서 벗어나 있습니다. 대체육에 대한 내용에만 초점을 맞추어, 콩고기에 대한 설명을 추가하거나 콩고기 이외에도 어떠한 식품들이 대체육에 해당하는지 서술하는 것이 좋을 것 같습니다.","문법적으로 약간의 수정이 필요합니다. '따라서 대체육은 고기가 아니다.' 또는 '결론적으로 대체육이 고기가 아님을 알 수 있다.' 등으로 바꿔주시는 것이 좋을 것으로 보입니다."]
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