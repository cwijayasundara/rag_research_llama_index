from dotenv import load_dotenv
import warnings
import nest_asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex

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

splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)

nodes = splitter.get_nodes_from_documents(documents)

# create index
index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What did Paul Graham do growing up?")
# print("query response", response)

# Summarization
from llama_index.core import SummaryIndex
summary_index = SummaryIndex(nodes)
query_engine = summary_index.as_query_engine()
summary = query_engine.query("Provide the summary of the document.")
# print("summary", summary)

# Chat engine
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
response = chat_engine.chat("What did Paul Graham do after YC?")
print("chat response: ", response)

response = chat_engine.chat("What about after that?")
print("chat response: ", response)

response = chat_engine.chat("Can you tell me more?")
print("chat response: ", response)

# CondenseContext ChatEngine
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    llm=llm,
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about an essay discussing Paul Grahams life."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=True,
)

response = chat_engine.chat("Hello")
print("chat response: ", response)