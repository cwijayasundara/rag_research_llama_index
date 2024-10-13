from dotenv import load_dotenv
import warnings
import nest_asyncio
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import (
    FunctionCallingAgentWorker,
    ReActAgent,
)
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
embed_model = OpenAIEmbedding()

Settings.llm = llm
Settings.embed_model = embed_model

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)

agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, subtract_tool], llm=llm, verbose=True
)

response = agent.chat("What is (26 * 2) + 2024?")
print(response)

# Agent with RAG Query Engine Tools
insurance_policy_docs = SimpleDirectoryReader(input_files=["../docs/policy/pb116349-business-health-select-handbook-1024-pdfa.pdf"]).load_data()

insurance_index = VectorStoreIndex.from_documents(insurance_policy_docs)
insurance_query_engine = insurance_index.as_query_engine(similarity_top_k=3)

response = insurance_query_engine.query("Whats the cashback option for dental fees?")
print("RAG response 1:", response)

response = insurance_query_engine.query("Whats the cashback option for optical expenses?")
print("RAG response 2:", response)

# FunctionCallingAgent with RAG QueryEngineTools.

query_engine_tools = [
    QueryEngineTool(
        query_engine=insurance_query_engine,
        metadata=ToolMetadata(
            name="insurance policy document",
            description="Provides information about insurance and claim policies",
        ),
    )
]

agent_worker = FunctionCallingAgentWorker.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)
agent = agent_worker.as_agent()

response = agent.chat("Whats the cashback option for dental fees?")

print("FunctionCallingAgentWorker response :", response)