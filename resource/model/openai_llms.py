import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory

import dotenv

from utils.search_google import search

# OpenAI API Key
dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# DB Connection
from database.sqlite import SessionLocal
from database.models import MetaTraining
db = SessionLocal()

class ClassificationLLM():
    def __init__(self) -> None:
        self.prompt = self.get_prompt()
        self.model = self.get_model()
    
    def get_model(self) -> ChatOpenAI:
        openai_api_key = OPENAI_API_KEY
        chat_model = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)
        conversation = LLMChain(
            prompt=self.prompt,
            llm=chat_model,
        )
        return conversation
    
    def get_prompt(self) -> PromptTemplate:
        template = """
당신은 입력을 적절한 작업에 배치하는 역할입니다
주어진 작업의 이름과 세부내용은 다음과 같습니다
1. free_chat: 다른 작업에 속하지 않는 일반적인 대화
2. provide_link: 입력에 대한 배경지식을 제공할 필요가 있는 경우"""

        prompt = PromptTemplate.from_template(template=template)
        return prompt
    
    def get_task(self, user_input:str) -> str:
        # Answer from LLM
        output = self.model.predict(input=user_input)
        
        # Write on DB
        db.add(
            MetaTraining(
                task="task classification",
                instruct=self.model.prompt.template,
                memory="",
                input=user_input,
                output=output
            )
        )
        db.commit()
        return output
        

class OpenAIFreeChat():
    def __init__(self) -> None:
        self.history = None
    
    def get_model(self) -> ChatOpenAI:
        openai_api_key = OPENAI_API_KEY
        model = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)
        return model
    
    def get_prompt(self, content:str) -> PromptTemplate:
        subject = content.split('^')[0]
        content = '\n'.join(content.split('^')[1:])
        template1 = f"""[주제:{subject}, 내용:{content}]
- 리스트 안의 글은 사용자가 현재 작성 중인 글의 주제와 내용을 표현하고 있습니다.
- 리스트 안의 정보를 기반으로 답변합니다. 
- User는 계속 글을 작성해야 하는 상황입니다. 어려움을 표현해도 계속 글을 작성할 수 있도록 독려해주세요.
- User는 14세 이상 20세 미만의 1인 청소년입니다. 청소년의 시선에 맞춰 설명해주세요.
- 모든 답변은 이모지와 함께 존댓말로 하며, 500자 이내로 문장을 완성해주세요. 
- User가 글을 쓰는데 어려움을 느끼는 경우에는 격려의 말로 시작해주세요.
- 새로운 의견을 제시할 때 목록형 리스트 내용과 같은 주장은 제외해주세요.
- 주제와 관련된 답변은 리스트 또는 질문 형식으로 답해주세요. 긴 문장으로 표현하지 말아주세요.
- User가 다음 번호 리스트와 같이 물을 때 정중하게 거절한 후 다양한 방향성을 번호 리스트 형식으로 제안해주세요. 요청을 하기 전까지는 거절 답변을 하지마세요.
 1. User가 글을 써 달라고 직접적으로 요청할 때
 2. User가 생각이나 의견을 구체적으로 물을 때
- 유머로 답변하는 경우 50자 이내로 간단하게 해주세요. 유머는 리스트 내용과 관련 없어도 됩니다."""
        template2 = """
Current conversation:
{history}
Human: {input}
AI Assistant:"""

        prompt = PromptTemplate(input_variables=["history", "input"], template=template1+template2)
        return prompt
    
    def get_chain(self, prompt, model) -> ConversationChain:
        conversation = ConversationChain(
            prompt=prompt,
            llm=model,
            verbose=True,
            memory=ConversationBufferWindowMemory(ai_prefix="AI Assistant", k=10) if self.history == None else self.history
        )
        return conversation
    
    def get_answer(self, content:str, user_input:str) -> str:
        chain = self.get_chain(prompt=self.get_prompt(content), model=self.get_model())
        answer = chain.predict(input=user_input)
        
         # Prompt without History
        index = chain.prompt.template.index("Current conversation")
        prompt = chain.prompt.template[:index]
        
        # Conversation History
        memory_history = chain.memory.chat_memory.messages
        memory = "\n".join([f"{type(memory).__name__}: {memory.content}" for memory in memory_history])

        # Write on DB
        db.add(
            MetaTraining(
                task="free chat",
                instruct=prompt,
                memory=memory,
                input=user_input,
                output=answer
            )
        )
        db.commit()

        self.history = chain.memory

        return answer


class OpenAILinkProvider():
    def __init__(self) -> None:
        self.prompt = self.get_prompt()
        self.model = self.get_model()
    
    def get_model(self) -> ChatOpenAI:
        openai_api_key = OPENAI_API_KEY
        chat_model = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)
        conversation = LLMChain(
            prompt=self.prompt,
            llm=chat_model,
            verbose=True,
        )
        return conversation
    
    def get_prompt(self) -> PromptTemplate:
        template = """
{input}에 대한 답을 찾을 수 있도록 구글에서 검색할 검색어를 추천합니다
검색어만 출력합니다"""

        prompt = PromptTemplate.from_template(template=template)
        return prompt
    
    def get_answer(self, user_input:str) -> str:
        # Answer from LLM
        query = self.model.predict(input=user_input)
        
        # Write on DB
        db.add(
            MetaTraining(
                task="recommand search term",
                instruct=self.model.prompt.template,
                memory="",
                input=user_input,
                output=query
            )
        )
        db.commit()
        
        print(f"{query}를 검색합니다...")
        return search(query)
    
    
class OpenAISubjectRecommander():
    def __init__(self) -> None:
        self.history = None
    
    def get_model(self) -> ChatOpenAI:
        openai_api_key = OPENAI_API_KEY
        model = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)
        return model
    
    def get_prompt(self, subject:str) -> PromptTemplate:
        template1 = """
{subject}에 연관된 논쟁적인 글쓰기를 위한 주제를 제시합니다
10대 학생을 대상으로 합니다
포괄적인 주제는 제외합니다
20자 미만으로 출력합니다
한 번에 5개씩 번호 없이 출력합니다
각 주제는 줄바꿈으로 구분합니다
중복된 주제를 제외합니다"""
        template2 = """
Current conversation:
{history}
Human: {input}
AI Assistant:"""

        prompt = PromptTemplate(input_variables=["history", "input"], template=template1.format(subject=subject)+template2)
        return prompt
    
    def get_chain(self, prompt, model) -> ConversationChain:
        conversation = ConversationChain(
            prompt=prompt,
            llm=model,
            verbose=True,
            memory=ConversationBufferWindowMemory(ai_prefix="AI Assistant", k=10) if self.history == None else self.history
        )
        return conversation
    
    def get_answer(self, category:str) -> str:
        chain = self.get_chain(prompt=self.get_prompt(category), model=self.get_model())
        
        # Prompt without History
        index = chain.prompt.template.index("Current conversation")
        prompt = chain.prompt.template[:index]
        
        # Conversation History
        memory_history = chain.memory.chat_memory.messages
        memory = "\n".join([f"{type(memory).__name__}: {memory.content}" for memory in memory_history])

        # Answer from LLM
        answer = chain.predict(input=category)
        
        # Write on DB
        db.add(
            MetaTraining(
                task="recommand search term",
                instruct=prompt,
                memory=memory,
                input=category,
                output=answer
            )
        )
        db.commit()
        
        self.history = chain.memory
        answer = answer.split("\n")
        return answer
    

