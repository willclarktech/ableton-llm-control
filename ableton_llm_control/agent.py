from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_agent_executor

DEFAULT_LLM_MODEL_NAME = "gpt-3.5-turbo-0125"
DEFAULT_PROMPT = hub.pull("hwchase17/openai-functions-agent")


def create_agent(
    tools: list[StructuredTool],
    model_name=DEFAULT_LLM_MODEL_NAME,
    prompt: ChatPromptTemplate = DEFAULT_PROMPT,
) -> CompiledGraph:
    llm = ChatOpenAI(model=model_name, temperature=0)
    agent_runnable = create_openai_functions_agent(llm, tools, prompt)
    return create_agent_executor(agent_runnable, tools)
