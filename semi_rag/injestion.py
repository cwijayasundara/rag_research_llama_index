import pickle
from dotenv import load_dotenv
import warnings
import nest_asyncio
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext

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

consolidated_nodes = text_nodes + index_nodes

# add chroma as the vector store
doc_store = SimpleDocumentStore()
doc_store.add_documents(consolidated_nodes)
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("dense_vectors")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(
    docstore=doc_store, vector_store=vector_store
)
# end chroma

recursive_index = VectorStoreIndex(nodes=text_nodes + index_nodes,
                                   storage_context=storage_context)

# tests

recursive_query_engine = recursive_index.as_query_engine(
    similarity_top_k=3, verbose=True
)

query = "What is the cashback option for dental expenses?"
response = recursive_query_engine.query(query)
print(
    "\n************New LlamaParse+ Recursive Retriever Query Engine************"
)
print(response)