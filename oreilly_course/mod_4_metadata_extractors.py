from dotenv import load_dotenv
import warnings
import nest_asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import MetadataMode
from llama_index.core import Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
)
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

llm = OpenAI(temperature=0.1, model="gpt-4o-mini",max_tokens=512)

Settings.llm = llm

# Node Parser and Metadata Extractors
node_parser = TokenTextSplitter(
    separator=" ", chunk_size=256, chunk_overlap=128
)

question_extractor = QuestionsAnsweredExtractor(
    questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
)

# load data
reader = SimpleWebPageReader(html_to_text=True)
docs = reader.load_data(urls=["https://eugeneyan.com/writing/llm-patterns/"])

orig_nodes = node_parser.get_nodes_from_documents(docs)

# build index
index0 = VectorStoreIndex(orig_nodes)
query_engine0 = index0.as_query_engine(similarity_top_k=1)

query_str = (
    "Can you describe metrics for evaluating text generation quality, compare"
    " them, and tell me about their downsides"
)

response0 = query_engine0.query(query_str)

print(response0)

# Extract Metadata Using PydanticProgramExtractor
from pydantic import BaseModel, Field
from typing import List

class NodeMetadata(BaseModel):
    """Node metadata."""

    entities: List[str] = Field(
        ..., description="Unique entities in this text chunk."
    )
    summary: str = Field(
        ..., description="A concise summary of this text chunk."
    )

from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.extractors import PydanticProgramExtractor

EXTRACT_TEMPLATE_STR = """\
Here is the content of the section:
----------------
{context_str}
----------------
Given the contextual information, extract out a {class_name} object.\
"""

openai_program = OpenAIPydanticProgram.from_defaults(
    output_cls=NodeMetadata,
    prompt_template_str="{input}",
    extract_template_str=EXTRACT_TEMPLATE_STR,
)

metadata_extractor = PydanticProgramExtractor(
    program=openai_program, input_key="input", show_progress=True
)

extract_metadata = metadata_extractor.extract(orig_nodes[0:1])

print(extract_metadata)
