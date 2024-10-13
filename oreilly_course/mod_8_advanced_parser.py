from dotenv import load_dotenv
import warnings
import nest_asyncio
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini")

Settings.llm = llm
Settings.embed_model = embed_model

# extract content from the PDF using LamaParse
documents = LlamaParse(result_type="markdown").load_data(
    "../docs/policy/pb116349-business-health-select-handbook-1024-pdfa.pdf"
)

# parse the content using MarkdownElementNodeParser
node_parser = MarkdownElementNodeParser(
    llm=OpenAI(model="gpt-4o-mini"), num_workers=8
)

nodes = node_parser.get_nodes_from_documents(documents)

text_nodes, index_nodes = node_parser.get_nodes_and_objects(nodes)

recursive_index = VectorStoreIndex(nodes=text_nodes + index_nodes)

raw_index = VectorStoreIndex.from_documents(documents)

# reranker
reranker = FlagEmbeddingReranker(
    top_n=3,
    model="BAAI/bge-reranker-large",
)

recursive_query_engine = recursive_index.as_query_engine(
    similarity_top_k=15, node_postprocessors=[reranker], verbose=True
)

raw_query_engine = raw_index.as_query_engine(
    similarity_top_k=15, node_postprocessors=[reranker]
)

query = "What is the cashback option for dental expenses?"

response_1 = raw_query_engine.query(query)

print("\n************New LlamaParse+ Basic Query Engine************")
print(response_1)

response_2 = recursive_query_engine.query(query)
print(
    "\n************New LlamaParse+ Recursive Retriever Query Engine************"
)
print(response_2)