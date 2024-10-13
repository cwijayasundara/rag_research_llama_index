from dotenv import load_dotenv
import warnings
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext

warnings.filterwarnings('ignore')
_ = load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini")

Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

documents = SimpleDirectoryReader("docs/essay").load_data()

splitter = SentenceSplitter(chunk_size=512)

nodes = splitter.get_nodes_from_documents(documents)

doc_store = SimpleDocumentStore()

doc_store.add_documents(nodes)

db = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = db.get_or_create_collection("dense_vectors")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(
    docstore=doc_store, vector_store=vector_store
)

index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

retriever = index.as_retriever(similarity_top_k=2)

query = "What did the author do after RISD?"

retrieved_nodes = retriever.retrieve(query)

for node in retrieved_nodes:
    print(node.text)