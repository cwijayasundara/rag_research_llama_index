from dotenv import load_dotenv
import warnings
import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

documents = SimpleDirectoryReader("../docs/essay/").load_data()

client = qdrant_client.QdrantClient(location=":memory:")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="llama_index_vector_store"
)


pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=1024, chunk_overlap=100),
        TitleExtractor(),
        OpenAIEmbedding(),
    ],
    vector_store=vector_store,
)
# Ingest directly into a vector db
nodes = pipeline.run(documents=documents)

index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine()

response = query_engine.query("What did paul graham do growing up?")

print(response)
