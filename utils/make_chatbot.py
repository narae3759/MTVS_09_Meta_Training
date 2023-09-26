import openai

# DB Connection
from database.sqlite import SessionLocal
from database.models import MetaTraining
db = SessionLocal()

def intention(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """
                            당신은 입력을 적절한 작업에 배치하는 역할입니다
                            주어진 작업의 이름과 세부내용은 다음과 같습니다
                            1. free_talk: 다른 작업에 속하지 않는 일반적인 대화
                            2. provide_link: 입력에 대한 배경지식을 제공할 필요가 있는 경우
                           """
            },
            {
                "role": "user",
                "content": f"{user_input}"
            }
        ],
        temperature=1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    result = response['choices'][0]['message']['content']
    
    system_text = """
당신은 입력을 적절한 작업에 배치하는 역할입니다
주어진 작업의 이름과 세부내용은 다음과 같습니다
1. free_talk: 다른 작업에 속하지 않는 일반적인 대화
2. provide_link: 입력에 대한 배경지식을 제공할 필요가 있는 경우
"""
    # Write on DB
    db.add(
        MetaTraining(
            task="make title",
            instruct=system_text,
            memory="",
            input=user_input,
            output=result
        )
    )
    db.commit()

    return result

def freechat(content, user_input):
    subject = content.split('^')[0]
    content = '\n'.join(content.split('^')[1:])

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"[주제:{subject}, 내용:{content}]\n- 리스트 안의 글은 사용자가 현재 작성 중인 글의 주제와 내용을 표현하고 있습니다.\n- 리스트 안의 정보를 기반으로 답변합니다. \n- User는 계속 글을 작성해야 하는 상황입니다. 어려움을 표현해도 계속 글을 작성할 수 있도록 독려해주세요.\n- User는 14세 이상 20세 미만의 1인 청소년입니다. 청소년의 시선에 맞춰 설명해주세요.\n- 모든 답변은 이모지와 함께 존댓말로 하며, 500자 이내로 문장을 완성해주세요.\n- User가 글을 쓰는데 어려움을 느끼는 경우에는 격려의 말로 시작해주세요.\n- 새로운 의견을 제시할 때 리스트 내용과 같은 주장은 제외해주세요.\n- 주제와 관련된 답변은 리스트 또는 질문 형식으로 답해주세요. 긴 문장으로 표현하지 말아주세요.\n- User가 다음 번호 리스트와 같이 물을 때 정중하게 거절한 후 다양한 방향성을 번호 리스트 형식으로 제안해주세요. 요청을 하기 전까지는 거절 답변을 하지마세요.\n 1. User가 글을 써 달라고 직접적으로 요청할 때\n 2. User가 생각이나 의견을 구체적으로 물을 때\n- 유머로 답변하는 경우 50자 이내로 간단하게 해주세요. 유머는 리스트 내용과 관련 없어도 됩니다."
            },
            {
                "role": "user",
                "content": f"{user_input}"
            }
        ],
        temperature=1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    result = response['choices'][0]['message']['content']
    
    # Write on DB
    db.add(
        MetaTraining(
            task="make title",
            instruct=f"[주제:{subject}, 내용:{content}]\n- 리스트 안의 글은 사용자가 현재 작성 중인 글의 주제와 내용을 표현하고 있습니다.\n- 리스트 안의 정보를 기반으로 답변합니다. \n- User는 계속 글을 작성해야 하는 상황입니다. 어려움을 표현해도 계속 글을 작성할 수 있도록 독려해주세요.\n- User는 14세 이상 20세 미만의 1인 청소년입니다. 청소년의 시선에 맞춰 설명해주세요.\n- 모든 답변은 이모지와 함께 존댓말로 하며, 500자 이내로 문장을 완성해주세요.\n- User가 글을 쓰는데 어려움을 느끼는 경우에는 격려의 말로 시작해주세요.\n- 새로운 의견을 제시할 때 리스트 내용과 같은 주장은 제외해주세요.\n- 주제와 관련된 답변은 리스트 또는 질문 형식으로 답해주세요. 긴 문장으로 표현하지 말아주세요.\n- User가 다음 번호 리스트와 같이 물을 때 정중하게 거절한 후 다양한 방향성을 번호 리스트 형식으로 제안해주세요. 요청을 하기 전까지는 거절 답변을 하지마세요.\n 1. User가 글을 써 달라고 직접적으로 요청할 때\n 2. User가 생각이나 의견을 구체적으로 물을 때\n- 유머로 답변하는 경우 50자 이내로 간단하게 해주세요. 유머는 리스트 내용과 관련 없어도 됩니다.",
            memory="",
            input=user_input,
            output=result
        )
    )
    db.commit()

    return result
