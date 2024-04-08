import streamlit as st 
from openai import OpenAI 
import os 
import numpy as np
from numpy.linalg import norm
import json

os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

if 'messages' not in st.session_state:
    st.session_state.messages = []

option = st.sidebar.selectbox(
    "설문 항목을 선택해주세요",
    ['기본 정보', '생활 습관 및 취미', '인생 목표 및 우선순위', '개인 가치관', '챗봇과 대화', '결과 종합']
)

if 'info_vectors' not in st.session_state:
    st.session_state.info_vectors = []
if 'wants' not in st.session_state:
    st.session_state.wants = []

if option =="기본 정보":
    st.header('기본 정보')
    with st.form(key="기본 정보"):
        name = st.text_input('이름을 입력해주세요.')
        age = st.number_input('나이를 입력해주세요.', step=1)
        gender = st.selectbox(
            '성별은 무엇인가요',
            ['남', '여']
        )
        area = st.text_input("거주지는 어디인가요?")
        job = st.text_input('직업 또는 직업 분야가 어떻게 되시나요? 학생이실 경우 전공이 어떻게 되시나요?')
        height = st.number_input("키를 입력해주세요")
        weight = st.number_input("몸무게를 입력해주세요.")
        photo = st.file_uploader("사진을 업로드해주세요.")
        profit = st.selectbox(
            '월수익이 어느정도 되시나요?',
            ['0~50', '50~100', '100~200', '200~400', '400~600', '600~800', '800 이상']
        )
        
        submit = st.form_submit_button(label='제출')
        
    if submit:
        SUMMARIZER_PROMPT = """
        You will be given the information about users. You have to summarize user's input with just two sentences. You have to include every information given by the user. You have to answer in Korean.
        """.strip()
        
        USER_INFO = f"""
        이름 : {name}
        나이 : {age}
        성 : {gender}
        거주지 : {area}
        직업, 전공 : {job}
        키, 몸무게 : {height}, {weight}
        수익 : {profit}
        """
        
        summary = client.chat.completions.create(
            model='gpt-3.5-turbo-0125',
            messages=[
                {'role': 'system', 'content': SUMMARIZER_PROMPT},
                {'role': 'user', 'content': USER_INFO}
            ]
        ).choices[0].message.content
        
        st.session_state.messages.append({'role': 'basic_info', 'content': summary})
        
        st.markdown(summary)

if option == "생활 습관 및 취미":
    st.header(option)
    with st.form(key=option):
        hobby = st.text_input("주요 취미나 관심사는 무엇입니까?")
        smoking = st.text_input("음주 및 흡연 습관에 대해 말씀해주세요!")
        smoking_date = st.text_input("흡연 및 음주하는 이성에 대해 어떻게 생각하시나요.")
        algorithm = st.text_input("TV 혹은 유튜브에서 주로 무엇을 보시나요?")
        
        submit = st.form_submit_button(label='제출')
    
    if submit:
        SUMMARIZER_PROMPT = """
        You will be given the information about users. You have to summarize user's input with two sentences. You have to include every information given by the user. You have to answer in Korean.
        """.strip()
        
        USER_INFO = f"""
        취미 : {hobby}
        흡연, 음주 습관 : {smoking}
        흡연 및 음주를 하는 이성에 대한 가치관 : {smoking_date}
        TV, 유튜브에서의 관심사 : {algorithm}
        """
        
        summary = client.chat.completions.create(
            model='gpt-3.5-turbo-0125',
            messages=[
                {'role': 'system', 'content': SUMMARIZER_PROMPT},
                {'role': 'user', 'content': USER_INFO}
            ]
        ).choices[0].message.content
        
        st.session_state.messages.append({'role': 'habit', 'content': summary})
        st.markdown(summary)

if option =="인생 목표 및 우선순위":
    st.header(option)
    with st.form(key=option):
        goal = st.text_input("인생에서의 장기적인 목표는 무엇인가요?")
        family = st.text_input("가족 계획에 대해 어떻게 생각하십니까?")
        career = st.text_input("경력에서 중요하게 생각하는 것은 무엇인가요? ex. 급여, 워라벨 등")
        
        submit = st.form_submit_button(label='제출')
    
    if submit:
        SUMMARIZER_PROMPT = """
        You will be given the information about users. You have to summarize user's input with two sentences. You have to include every information given by the user. You have to answer in Korean.
        """.strip()
        
        USER_INFO = f"""
        인생의 장기적 목표 : {goal}
        가족 계획 : {family}
        경력에서 중요하게 생각하는 것 : {career}
        """
        
        summary = client.chat.completions.create(
            model='gpt-3.5-turbo-0125',
            messages=[
                {'role': 'system', 'content': SUMMARIZER_PROMPT},
                {'role': 'user', 'content': USER_INFO}
            ]
        ).choices[0].message.content
        
        st.session_state.messages.append({'role': 'goal', 'content': summary})
        st.markdown(summary)

if option == "개인 가치관":
    st.header(option)
    with st.form(key=option):
        happiest = st.text_input("가장 행복한 순간은 언제였나요?")
        future_happy = st.text_input("미래의 행복을 위해 가장 중요한 것은 무엇인가요?")
        criteria = st.text_input("당신에게 가장 중요한 가치 세가지는 무엇인가요?")
        
        submit = st.form_submit_button(label='제출')
    
    if submit:
        SUMMARIZER_PROMPT = """
        You will be given the information about users. You have to summarize user's input with two sentences. You have to include every information given by the user. You have to answer in Korean.
        """.strip()
        
        USER_INFO = f"""
        인생의 가장 행복한 순간 : {happiest}
        미래의 행복을 위해 가장 중요한 요소 : {future_happy}
        가장 중요한 가치 세가지 : {criteria}
        """
        
        summary = client.chat.completions.create(
            model='gpt-3.5-turbo-0125',
            messages=[
                {'role': 'system', 'content': SUMMARIZER_PROMPT},
                {'role': 'user', 'content': USER_INFO}
            ]
        ).choices[0].message.content
        
        st.session_state.messages.append({'role': 'criteria', 'content': summary})


if option =="챗봇과 대화":
    sex = st.select_slider("성별", ["남성", "여성"])
    questions = "이성에게서 가장 중요시하는 가치는 무엇인가요?"
    QUESTION_GENERATOR = """
    You're tasked with identifying qualities a user seeks in a partner, drawing from our message exchanges. Your goal is to ask precise, clear questions to uncover these qualities. Avoid vague or complex inquiries. Use follow-up questions to probe deeper into the user's answers. Your approach should be methodical. A successful interview will earn you a $10 reward. Failure to meet expectations will result in consequences. Responses should be **in Korean**.
    """.strip()
    
    VALIDATOR = """
    From the user's response, you need to determine what values ​​the user considers important in the opposite sex. Your response should be one of basic_info, habit, goal, criteria. basic_info refers to information such as height, weight, appearance, etc. of the opposite sex. Habit refers to a habit that the opposite sex usually has. Goal refers to the long-term or short-term goals that the opposite sex has in life, and criteria refers to the values ​​of the opposite sex. You must answer one of the four items above.
    
    Your answer should be in the form of json object with following format:
    {"quality": "Value the user considers in the opposite sex. Either basic_info, habit, goal or criteria", "reason": "reason why you thought so."}
    """
    
    if "context" not in st.session_state:
        st.session_state.context = []
    if 'thread' not in st.session_state:
        st.session_state.thread = []
    if 'initialize' not in st.session_state:
        st.session_state.initialize = False
    
    def initialize():
        st.session_state.initialize = True
    
    if not st.session_state.initialize:
        st.session_state.context.append({'role': 'system', 'content': QUESTION_GENERATOR})    
        st.session_state.context.append({'role': 'assistant', 'content': questions})
        st.session_state.thread.append({'role': 'assistant', 'content': questions})
    
    button = st.button("시작하기", on_click=initialize)
    

    for message in st.session_state.thread:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    answer = st.chat_input("질문에 답해주세요!")
    if answer:
        st.session_state.context.append({'role': 'user', 'content': answer})
        st.session_state.thread.append({'role': 'user', 'content': answer})
        with st.chat_message('user'):
            st.markdown(answer)
        ques = client.chat.completions.create(
            model='gpt-3.5-turbo-0125',
            messages=st.session_state.context
        ).choices[0].message.content
        st.session_state.context.append({'role': 'assistant', 'content': ques})
        st.session_state.thread.append({'role': 'assistant', 'content': ques})
        with st.chat_message('assistant'):
            st.markdown(ques)
    
    if 'basic_info' not in st.session_state:
        st.session_state.basic_info = []
    if 'habit' not in st.session_state:
        st.session_state.habit = []
    if 'goal' not in st.session_state:
        st.session_state.goal = []
    if 'criteria' not in st.session_state:
        st.session_state.criteria = []
    
    collected_basic = ""
    collected_habit = ''
    collected_goal = ''
    collected_criteria = ''
    if len(st.session_state.thread) > 8:
        st.spinner("정보가 처리되고 있습니다!")
        for message in st.session_state.thread:
            validation = client.chat.completions.create(
                model='gpt-4-turbo-preview',
                messages=[
                    {"role": "system", "content": VALIDATOR},
                    {'role': 'user', 'content': message['content']}
                ],
                response_format={'type': 'json_object'}
            ).choices[0].message.content
            
            json_validaiton = json.loads(validation)
            if json_validaiton['quality'] == 'basic_info':
                collected_basic += message['content'] + '\n'
            elif json_validaiton['quality'] == 'habit':
                collected_habit += message['content'] + '\n'
            elif json_validaiton['quality'] == 'goal':
                collected_goal += message['content'] + '\n'
            elif json_validaiton['quality'] == 'criteria':
                collected_criteria += message['content'] + '\n'
                
        SUMMARIZER_PROMPT = """
        You will be given the information about qualities user seeks in a partner. You have to summarize user's qualities with three sentences. You have to include **every information** given by the user. You have to answer in Korean.
        """.strip()
        
        summary_basic = client.chat.completions.create(
            model='gpt-3.5-turbo-0125',
            messages=[
            {'role': 'system', 'content': SUMMARIZER_PROMPT},
            {'role': 'user', 'content': collected_basic}
            ]
        ).choices[0].message.content
        
        summary_habit = client.chat.completions.create(
            model='gpt-3.5-turbo-0125',
            messages=[
                {'role': 'system', 'content': SUMMARIZER_PROMPT},
                {'role': 'user', 'content': collected_habit}
            ]
        ).choices[0].message.content 
        
        summary_goal = client.chat.completions.create(
            model='gpt-3.5-turbo-0125',
            messages=[
                {'role': 'system', 'content': SUMMARIZER_PROMPT},
                {'role': 'user', 'content': collected_goal}
            ]
        ).choices[0].message.content
        
        summary_criteria = client.chat.completions.create(
            model='gpt-3.5-turbo-0125',
            messages=[
                {'role': 'system', 'content': SUMMARIZER_PROMPT},
                {'role': 'user', 'content': collected_criteria}
            ]
        ).choices[0].message.content
        
        st.session_state.basic_info.append(summary_basic + '\n')
        st.session_state.habit.append(summary_habit + '\n')
        st.session_state.goal.append(summary_goal + '\n')
        st.session_state.criteria.append(summary_criteria + '\n')
        
        st.markdown(f"이성의 기본 정보에 대한 희망 사항: {summary_basic}")
        st.markdown(f"이성의 습관에 대한 희망 사항: {summary_habit}")
        st.markdown(f"이성의 목표에 대한 희망 사항: {summary_goal}")
        st.markdown(f"이성의 가치관에 대한 희망 사항: {summary_criteria}")

        
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def cossim(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    
    # Compute the L2 norms (Euclidean norms) of vector_a and vector_b
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    # Compute the cosine similarity
    similarity = dot_product / (norm_a * norm_b)
    
    return similarity

if option == "결과 종합":
    for message in st.session_state.messages:
        vector = get_embedding(message['content'])
        st.session_state.info_vectors.append({'role': message['role'], 'content': vector})
    
    for basic, habit, goal, criteria in zip(st.session_state.basic_info, st.session_state.habit, st.session_state.goal, st.session_state.criteria):
        basic_vector = get_embedding(basic[0])
        habit_vector = get_embedding(habit[0])
        goal_vector = get_embedding(goal[0])
        criteria_vector = get_embedding(criteria[0])
    
    for message in st.session_state.info_vectors:
        if message['role'] == 'basic_info':
            sim = cossim(message['content'], basic_vector)
        if message['role'] == 'habit':
            sim = cossim(message['content'], habit_vector)
        if message['role'] == 'goal':
            sim = cossim(message['content'], goal_vector)
        if message['role'] == 'criteria':
            sim = cossim(message['content'], criteria_vector)
        
        with st.chat_message("assistant"):
            st.markdown(f"{message['role']}: {np.round(sim*200, 2)}")
