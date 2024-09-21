import base64
import os
import time
import re
import ast
import streamlit as st
from operator import itemgetter
from langchain.tools import tool
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.tools.render import render_text_description
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory
from langchain.chains import ConversationChain
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 设置背景图像 #######################################
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
img_path = "D:\桌面文件\学习\图片素材\结束图片.jpg"
img_base64 = get_base64_of_bin_file(img_path)
st.markdown(
    f"""  
    <style>  
    .stApp {{  
        background-image: url("data:image/jpeg;base64,{img_base64}");  
        background-size: cover;  
        background-repeat: no-repeat;  
        background-attachment: fixed;  
    }}  
    </style>  
    """,
    unsafe_allow_html=True
)
####################################################

st.title("First Terminal")

# 选择交流模式 (RAG 或 None_RAG)
mode = st.sidebar.selectbox("Choose Your Mode",("Normal","RAG","ReAct(test)"))

# 设置记忆清除按钮
bool_clear_memory = st.button("Start a new dialogue and del chat_history",type="primary")

# 设置温度选择 (这里的 label 和 options 是必要的)
temperature_choosen = st.sidebar.select_slider(label="Temperature",options=[i/10 for i in range(0,11)])
llama_model = OllamaLLM(model='llama3.1:8b',temperature=temperature_choosen)

# 解析器设置
json_output_parser = JsonOutputParser()
str_output_parser = StrOutputParser()

@st.cache_resource(ttl='1h')
def get_retriever(uploaded_file,embeddings):
    '''这里的的embedding有两种选择，分别是llama3.1:8b,qwen2:7b
    用str表述就可以了'''
    docs = []
    for file in uploaded_file:
        temp_filepath = os.path.join(temp_dir,file.name)
        with open(temp_filepath,"wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", ",", " ", ""]
    )
    x = recursive_splitter.split_documents(docs)

    if embeddings == "llama3.1:8b":
        chosen_embedding = OllamaEmbeddings(model="llama3.1:8b")
    elif embeddings == "qwen2:7b":
        chosen_embedding = OllamaEmbeddings(model="qwen2:7b")
    else:
        chosen_embedding = None

    vectordb = Chroma.from_documents(documents=x, embedding=chosen_embedding)  # 这里不保存到本地
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
    return retriever

def get_session_history(session_id: str):
    if session_id not in st.session_state["chat_history"]:
        st.session_state["chat_history"][session_id] = ChatMessageHistory()
    return st.session_state["chat_history"][session_id]
# 记忆总结链(用于 ReAct)
text_summary = """Here's the chat history between Human and AI assistant.
{history_inputs}
Please make a summary for it in the following format:\n
Human: <summary of human talk>
AI assistant: <summary of AI talk>
Human: ...
AI: ...(repeat this mode)"""
prompt_summary = ChatPromptTemplate.from_template(text_summary)
chain_summary = prompt_summary | llama_model

# 提取函数
def extract_functions_with_tool_decorator(code):
    """提取代码中的函数定义，并添加 @tool 装饰器"""
    tree = ast.parse(code)
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # 添加 @tool 装饰器
            tool_decorator = ast.Name(id='tool', ctx=ast.Load())
            node.decorator_list.insert(0, tool_decorator)
            func_code = ast.unparse(node)
            functions[node.name] = func_code
    return functions

# 提取单步 ReAct动作
def get_thoughts(x):
    '''获取到每一步的thoughts'''
    y = re.search(r"Thought:(.*?)\n",x,re.DOTALL)
    return y.group(1).strip()

def get_action(x):
    '''获取到每一步的action'''
    y = re.search(r"Action:(.*?)\n",x,re.DOTALL)
    return y.group(1).strip()

def get_action_input(x):
    '''获取到每一步action的参数'''
    y = re.search(r"Action Input:(.*?)}",x,re.DOTALL)
    yy = y.group(1).strip() + '}'
    yy = yy.replace("'", '"')
    return json_output_parser.parse(yy)

# 执行 ReAct
def check_if_finished(res:str):
    if res.find(r"Action Input") == -1:
        return False
    else:
        return True
def run_react_agent(user_input:str, prompt_react ,max_steps=8):
    flag = True # true表示这里没有结束ReAct过程
    one_chat_history = [] # 储存("human“,...)/("ai",...)的数据
    temp_ai_his = []
    one_chat_history.append(("user",user_input))
    chain_execute = {"inputs" : itemgetter("inputs"),
                    "history": itemgetter("history")
                     } | prompt_react | llama_model | str_output_parser
    response = chain_execute.invoke({"inputs":user_input,
                                     "history":st.session_state["chat_history_ReAct"].messages})

    temp_ai_his.append(response)
    flag = check_if_finished(response)
    while flag:
        tep_x = reshape_prompt_human(prompt_react.messages[2].prompt.template,response)
        prompt_react = ChatPromptTemplate.from_messages(
            [
                ('system',prompt_react.messages[0].prompt.template),
                MessagesPlaceholder(variable_name='history'),
                ('human',tep_x)
            ]
        )
        chain_execute = {"inputs" : itemgetter("inputs"),
                        "history": itemgetter("history")
                        } | prompt_react | llama_model | str_output_parser
        response = chain_execute.invoke({"inputs":user_input,
                                         "history":st.session_state["chat_history_ReAct"].messages})
        temp_ai_his.append(response)
        flag = check_if_finished(response)
    one_chat_history.append(("ai",'\n'.join(temp_ai_his)))
    return one_chat_history

# 更新 Prompt_ReAct
def reshape_prompt_human(x,response): # x是原来的 prompt
    the_thoughts = get_thoughts(response)
    the_action = get_action(response)
    the_action_input = get_action_input(response)
    observation = globals()[the_action].invoke(the_action_input)
    k = f"\nThought: {the_thoughts}\n" + f"Action: {the_action}\n" + "Action Input: {"+f"{the_action_input}" + "}\n" + \
    f"Observation: The output of the action is {observation}"
    prompt_new = x + k
    return prompt_new
# 重试函数
def retry_on_failure(func, user_input ,prompt_react,max_attempts=5, delay=0.5):
    attempts = 0
    while attempts < max_attempts:
        try:
            ans = func(user_input,prompt_react=prompt_react)
            return ans# 如果函数运行成功，直接返回
        except Exception as e:
            attempts += 1
            print(f"执行失败: {e}，重试 {attempts}/{max_attempts}")
            time.sleep(delay)  # 等待一段时间后重试
    print("超过最大重试次数，仍然失败。")


# 记忆设置
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}
if "chat_history_non_RAG" not in st.session_state:
    st.session_state["chat_history_non_RAG"] = ConversationBufferMemory()
if "chat_history_ReAct" not in st.session_state:
    st.session_state["chat_history_ReAct"] = ChatMessageHistory() # 通过这个实例的.messages即可获得列表
    st.session_state["chat_history_ReAct"].add_ai_message("Hello! How can I help you.")

if mode == "RAG":
    # 依据用户需求做记忆删除
    if bool_clear_memory:
        st.session_state["chat_history"] = {}

    # 加载文件
    upload_file = st.file_uploader(label="Your PDF Please", accept_multiple_files=True)
    temp_dir = st.sidebar.text_input(label="Your Temporary File Directory (For RAG)")

    # 做一个 button 用于清理现有的 file browser 以重新产生 retriever
    bool_change_retriever = st.button("Change Retriever", type="primary")

    if bool_change_retriever:
        st.cache_resource.clear()
        upload_file = None

    embeddings = st.sidebar.selectbox(
        "Choose Your Embedding Model for RAG",
        ("llama3.1:8b", "qwen2:7b")
    )

    text_sys = """You're a kind AI asistant, please chat with users and answer their question.
    Here will be some information about the question. If none of them is helpful for your problem solving,
    please use your own knowledge to solve user's problems.
    But for the question you don't know, please honestly say 'I don't know.'
    \n<context>
    {context}
    </context>"""
    text_human = """{inputs}"""

    prompts_RAG = ChatPromptTemplate.from_messages(
        [
            ("system",text_sys),
            MessagesPlaceholder(variable_name="history"),
            ("human",text_human)
        ]
    )

    if upload_file:
        session_id = "rag_01"
        retriever = get_retriever(upload_file,embeddings)

        sub_chain = {"context": itemgetter("inputs") | retriever,
                     "inputs": itemgetter("inputs"),
                     "history": itemgetter("history")} | prompts_RAG | llama_model

        conversation_RAG = RunnableWithMessageHistory(
            sub_chain,
            get_session_history,
            input_messages_key="inputs",
            history_messages_key="history"
        )

        #########
        # st.write(st.session_state["chat_history"])

        if st.session_state["chat_history"] != {}:
            for i in st.session_state["chat_history"][session_id].messages:
                st.chat_message(i.type).write(i.content)

        if user_input := st.chat_input():  # 这里设置为只有用户输入才做出反应
            st.chat_message("user").write(user_input)
            res = conversation_RAG.invoke({"inputs":user_input},
                                          config={"configurable": {"session_id": session_id}})
            st.chat_message("ai").write(res)

elif mode == "Normal":
    # 依据用户需求做记忆删除
    if bool_clear_memory:
        st.session_state["chat_history_non_RAG"] = ConversationBufferMemory()

    conversation = ConversationChain(
        llm=llama_model,
        memory=st.session_state["chat_history_non_RAG"]  # 将记忆导入
    )

    for i in conversation.memory.chat_memory.messages:  # 将数据读入，以展示以往的对话记录
        st.chat_message(i.type).write(i.content)  # 这里的 i.type 和 i.content 可以自己用jupyter试一下

    if user_input := st.chat_input():  # 这里设置为只有用户输入才做出反应
        st.chat_message("user").write(user_input)
        res = conversation.invoke(user_input)
        st.chat_message("ai").write(res["response"])


# ReAcT(先实现在非 RAG 下的多轮 ReAct 对话)
elif mode == "ReAct(test)":
    if bool_clear_memory:
        st.session_state["chat_history_ReAct"] = ChatMessageHistory()
        st.session_state["chat_history_ReAct"].add_ai_message("Hello! How can I help you.")
    uploaded_pyfile = st.file_uploader("选择一个Python文件", type="py")
    flag = True
    tools_info_list = [] # 这是一个str的列表，其中为函数的描述
    tools_name = [] # 这是函数的名字列表
    if uploaded_pyfile is not None:
        file_content = uploaded_pyfile.read().decode("utf-8") # 读取文件内容
    else:
        flag = False

    if flag:
        functions = extract_functions_with_tool_decorator(file_content)
        for func_name, func_code in functions.items():
            exec(func_code, globals()) # 在全局命名空间中执行函数定义,只有这个执行才能调用函数
            # 如果要调用就用 globals()[函数名:str].invoke(列表)
            tools_info_list.append(render_text_description([globals()[func_name]])) # 储存描述
            tools_name.append(func_name)

    tools_info = '\n'.join(tools_info_list) # 得到函数描述

    action_format = """
    {{
        arguments : {{...}}
    }}
    """
    prompt_sys = f"""Answer the following questions as best you can. You have access to the following tools:

    {tools_info}"""
    prompt_human = f"""For user's question, you need to follow the following resoning path:

    You will get some information about the previous resoning process. Based on them,
    please figure out what's the next ONE step, and return your answer in the following format:

    Thought: you should always think about what to do
    Action: the action to take, should be one of {tools_name}
    Action Input: the input parameters of the action.(In the format of Json blob)

    If you find you already got the answer from the Observation, then instead using the format above, please use the following format:

    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Attention! Every time you just need to output ONE STEP of 'Thought/Action/Action Input' or 'Thought/Final answer'.
    Don't perform multi-step reasoning. And if action exists, please output 'Action Inputs' in the following format(Json blob):
    {action_format}

    Now Let's Begin!
    Here're the Question and previous resoning process""" + """
    Question: {inputs}"""
    prompt_react = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_sys),
            MessagesPlaceholder(variable_name="history"), # 同 RAG中多轮对话实现
            ("human", prompt_human)
        ]
    )
    if st.session_state["chat_history_ReAct"].messages != []:
        for i in st.session_state["chat_history_ReAct"].messages:
            st.chat_message(i.type).write(i.content)

    if x := st.chat_input():
        st.chat_message('user').write(x)
        ans = retry_on_failure(run_react_agent, x,prompt_react)
        st.chat_message('ai').write(ans[1][1])
        st.session_state["chat_history_ReAct"].add_user_message(ans[0][1])
        st.session_state["chat_history_ReAct"].add_ai_message(ans[1][1])







