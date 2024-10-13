import nest_asyncio
from dotenv import load_dotenv
import warnings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini")
Settings.llm = llm

documents_with_instruction = (LlamaParse(
    result_type="markdown",
    parsing_instruction= """
                            This document is an insurance policy.
                            When a benefits/coverage/exclusion is describe in the document amend to it add a text in the 
                            following benefits string format (where coverage could be an exclusion).
                            For {nameofrisk} and in this condition {whenDoesThecoverageApply} the coverage is 
                            {coverageDescription}. 
                            If the document contain a benefits TABLE that describe coverage amounts, do not ouput it as 
                            a table, but instead as a list of benefits string.""")
                              .load_data("docs/policy/pb116349-business-health-select-handbook-1024-pdfa.pdf"))

node_parser = MarkdownElementNodeParser(
    llm=OpenAI(model="gpt-4o-mini"), num_workers=8
)

node_parser_instruction = MarkdownElementNodeParser(
    llm=OpenAI(model="gpt-4o-mini"), num_workers=8
)

nodes_instruction = node_parser.get_nodes_from_documents(documents_with_instruction)
(
    base_nodes_instruction,
    objects_instruction,
) = node_parser_instruction.get_nodes_and_objects(nodes_instruction)


recursive_index_instruction = VectorStoreIndex(
    nodes=base_nodes_instruction + objects_instruction
)

query_engine_instruction = recursive_index_instruction.as_query_engine(
    similarity_top_k=2
)

query_1 = "Whats the cash back option for dentist visits?"

print("With instructions:")
response_1_i = query_engine_instruction.query(query_1)
print(response_1_i)



