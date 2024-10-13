from dotenv import load_dotenv
import warnings
import nest_asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
embed_model = OpenAIEmbedding()

Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader(
    input_files=["../docs/essay/paul_graham_essay.txt"]
).load_data()

# create nodes
splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)

# Construct an index by loading documents into a VectorStoreIndex.
index = VectorStoreIndex(nodes)

# configure retriever
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

# configure response synthesizer
synthesizer = get_response_synthesizer(response_mode="refine")

# construct query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
)

response = query_engine.query("What did Paul Graham do growing up?")
print(response)

# index as a retriever
retriever = index.as_retriever(similarity_top_k=3)
retrieved_nodes = retriever.retrieve("What did Paul Graham do growing up?")

for text_node in retrieved_nodes:
    print(text_node)


