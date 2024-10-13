from dotenv import load_dotenv
import warnings
import nest_asyncio
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core.tools.query_engine import QueryEngineTool, ToolMetadata

from IPython.display import display, HTML

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

lyft_docs = SimpleDirectoryReader(
    input_files=["../data/10k/lyft_2021.pdf"]
).load_data()
uber_docs = SimpleDirectoryReader(
    input_files=["../data/10k/uber_2021.pdf"]
).load_data()

print(f"Loaded lyft 10-K with {len(lyft_docs)} pages")
print(f"Loaded Uber 10-K with {len(uber_docs)} pages")

lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

async def lyft_query(query: str):
    return await lyft_engine.aquery(query)

response = lyft_query("What is the revenue of Lyft in 2021? Answer in millions with page reference")

display(HTML(f'{response}'))

# Define QueryEngine Tools
query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description="Provides information about Lyft financials for year 2021",
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description="Provides information about Uber financials for year 2021",
        ),
    ),
]

from llama_index.core.query_engine.sub_question_query_engine import (
    SubQuestionQueryEngine,
)

sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)

async def sub_question_query(query: str):
    return await sub_question_query_engine.aquery(query)

response = sub_question_query(
    "Compare revenue growth of Uber and Lyft from 2020 to 2021"
)

async def main():
    response = await sub_question_query_engine.aquery(
        "Compare revenue growth of Uber and Lyft from 2020 to 2021"
    )
    display(HTML(f'{response}'))

import asyncio
asyncio.run(main())