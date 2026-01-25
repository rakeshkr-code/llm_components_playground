# # Component 1: Models with Ollama
from langchain_ollama import ChatOllama
from pprint import pprint

# Use local Llama (FREE, no API key!) [web:145]
llm = ChatOllama(
    model="llama3.2:1b",  # Smallest, fastest
    temperature=0.7
)

# Or use DeepSeek (better for code) [web:153]
llm_code = ChatOllama(
    model="deepseek-r1:1.5b"
)

# # Chat
# # response = llm.invoke("What is machine learning?")
# response = llm.invoke("Capital of India?")

# print(type(response), end="\n\n")
# pprint(dict(response))
# print()
# print(response.content)


# --------------------------------------------------------------

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("user", "{question}")
])
# print("PROMPT------")
# print(type(prompt))
# # print(prompt.format_prompt(
# #     role="Python expert",
# #     question="How to read JSON?"
# # ).to_string())
# pprint(dict(prompt))
# print()

# # Use with Ollama
# chain = prompt | llm | StrOutputParser()
# print("CHAIN------")
# print(type(chain))
# pprint(dict(chain))
# print()

# result = chain.invoke({
#     "role": "Geography teacher",
#     "question": "What is the capital of India?"
# })
# print("RESULT------")
# print(type(result))
# print(result)

# --------------------------------------------------------------

# # Component 3: LCEL with Local LLM

# # Multi-step chain (all local, FREE!)
# chain = (
#     prompt 
#     | llm  # Local Llama
#     | StrOutputParser()
# )

# # Parallel execution with multiple local models
# from langchain_core.runnables import RunnableParallel

# parallel = RunnableParallel(
#     llama_response=llm,           # Llama 3.2
#     deepseek_response=llm_code,   # DeepSeek
# )

# results = parallel.invoke([{"role": "user", "content": "Hello"}])

# print("PARALLEL RESULTS------")
# # pprint({k: v.content for k, v in results[0].items()})
# pprint(dict(results))

# --------------------------------------------------------------

# # Component 4: Memory
# from langchain_ollama import ChatOllama
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from pprint import pprint

# # Initialize
# llm = ChatOllama(model="llama3.2:1b")

# # Manual message history (simple & clear)
# messages = [
#     SystemMessage(content="You are a helpful Python tutor.")
# ]

# print("INITIAL STATE------")
# print(f"Messages: {len(messages)}")
# print()

# # Turn 1
# messages.append(HumanMessage(content="I'm learning Python"))
# response1 = llm.invoke(messages)
# messages.append(response1)
# print("Response 1:", response1.content)
# print()

# # Turn 2
# messages.append(HumanMessage(content="What should I learn first?"))
# response2 = llm.invoke(messages)
# messages.append(response2)
# print("Response 2:", response2.content)
# print()

# # Turn 3
# messages.append(HumanMessage(content="Can you summarize our conversation?"))
# response3 = llm.invoke(messages)
# messages.append(response3)
# print("FINAL RESULT------")
# print("Response 3:", response3.content)
# print()

# print("CONVERSATION HISTORY------")
# for i, msg in enumerate(messages):
#     print(f"{i+1}. {msg.type}: {msg.content[:80]}...")

# --------------------------------------------------------------

# # Component 5: Agents with Local Models

# from langchain.agents import create_openai_functions_agent, AgentExecutor
# from langchain.tools import tool

# # Define tools
# @tool
# def search_docs(query: str) -> str:
#     """Search local documentation."""
#     return f"Found docs about: {query}"
# @tool
# def calculator(query: str) -> str:
#     """Perform calculations."""
#     try:
#         result = eval(query)
#         return f"The result of {query} is {result}"
#     except Exception as e:
#         return f"Error in calculation: {e}"

# tools = [search_docs, calculator]

# # Agent with local Ollama [web:144]
# agent = create_openai_functions_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools)

# # Run (100% local, 100% FREE!)
# response = agent_executor.invoke({
#     "input": "Search for Python tutorials and calculate 10 * 5"
# })

# --------------------------------------------------------------
# from langchain_ollama import ChatOllama
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain_core.prompts import PromptTemplate
# from langchain_core.tools import tool

# print("Initializing Ollama...")

# # Initialize Ollama
# llm = ChatOllama(
#     model="llama3.2:1b",
#     temperature=0
# )

# # Define tools
# @tool
# def search_docs(query: str) -> str:
#     """Search local documentation about programming topics."""
#     return f"Found Python tutorial documentation about: {query}"

# @tool
# def calculator(expression: str) -> str:
#     """Perform mathematical calculations. Input should be like '10 * 5' or '2 + 2'."""
#     try:
#         result = eval(expression, {"__builtins__": {}}, {})
#         return str(result)
#     except Exception as e:
#         return f"Error: {str(e)}"

# # Tools list
# tools = [search_docs, calculator]

# # Create ReAct prompt template
# prompt_template = """Answer the following questions as best you can. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!

# Question: {input}
# Thought:{agent_scratchpad}"""

# prompt = PromptTemplate.from_template(prompt_template)

# # Create ReAct agent
# agent = create_react_agent(llm, tools, prompt)

# # Create agent executor
# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True,
#     handle_parsing_errors=True,
#     max_iterations=5
# )

# # Run the agent
# print("\n" + "="*60)
# print("Running Agent...")
# print("="*60 + "\n")

# try:
#     response = agent_executor.invoke({
#         "input": "Search for Python tutorials and calculate 10 * 5"
#     })
    
#     print("\n" + "="*60)
#     print("FINAL RESULT:")
#     print("="*60)
#     print(response['output'])
    
# except Exception as e:
#     print(f"\nAgent Error: {e}")
#     print("\nFalling back to direct tool usage...")
    
#     # Direct tool usage as fallback
#     doc_result = search_docs.invoke("Python tutorials")
#     calc_result = calculator.invoke("10 * 5")
    
#     print(f"\nDocumentation: {doc_result}")
#     print(f"Calculation: {calc_result}")
# --------------------------------------------------------------

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool

print("Initializing agent...")

# Initialize Ollama model
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

# Define tools (simple functions with docstrings)
@tool
def search_docs(query: str) -> str:
    """Search local documentation for programming topics."""
    return f"Found Python tutorials about: {query}. Topics include variables, functions, loops, and OOP."

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Input should be a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is {result}"
    except Exception as e:
        return f"Calculation error: {e}"

# Create agent (following documentation pattern)
agent = create_agent(
    model=llm,  # Use Ollama model instead of model name string
    tools=[search_docs, calculator],
    system_prompt="You are a helpful programming assistant with access to documentation search and calculator tools."
)

# Run the agent
print("\n" + "="*60)
print("Running agent...")
print("="*60 + "\n")

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Search for Python tutorials and calculate 10 * 5"}]}
)

# Print result
print("\n" + "="*60)
print("FINAL RESULT:")
print("="*60)
print(response["messages"][-1].content)
