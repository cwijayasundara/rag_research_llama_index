from dotenv import load_dotenv
import warnings
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
)
import logging
import sys
from llama_index.core import SimpleKeywordTableIndex
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import (
    LLMSingleSelector,
)
from llama_index.core.selectors.pydantic_selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)

warnings.filterwarnings('ignore')
_ = load_dotenv()

# Set up the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set logger level to INFO

# Clear out any existing handlers
logger.handlers = []

# Set up the StreamHandler to output to sys.stdout (Colab's output)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)  # Set handler level to INFO

# Add the handler to the logger
logger.addHandler(handler)

documents = SimpleDirectoryReader("../docs/essay/").load_data()

# create parser and parse document into nodes
parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
nodes = parser(documents)

# Create VectorStoreIndex and SummaryIndex.
# Summary Index for summarization questions
summary_index = SummaryIndex(nodes)

# Vector Index for answering specific context questions
vector_index = VectorStoreIndex(nodes)

# Summary Index Query Engine
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)

# Vector Index Query Engine
vector_query_engine = vector_index.as_query_engine()

# Build summary index and vector index tools
# Summary Index tool
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to Paul Graham eassy on What I Worked On.",
)

# Vector Index tool
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context from Paul Graham essay on What I Worked On.",
)

# PydanticSingleSelector
query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
)

response = query_engine.query("What is the summary of the document?")
print("Router response 1:", response)

# Create Router Query Engine
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
)
response = query_engine.query("What is the summary of the document?")
print("Router response 2:", response)

# PydanticMultiSelector

keyword_index = SimpleKeywordTableIndex(nodes)

keyword_query_engine = keyword_index.as_query_engine()

keyword_tool = QueryEngineTool.from_defaults(
    query_engine=keyword_query_engine,
    description="Useful for retrieving specific context using keywords from Paul Graham essay on What I Worked On.",
)

query_engine = RouterQueryEngine(
    selector=PydanticMultiSelector.from_defaults(),
    query_engine_tools=[vector_tool, keyword_tool, summary_tool],
)

response = query_engine.query(
    "What were noteable events and people from the authors time at Interleaf and YC?"
)
print("Router response 3:", response)