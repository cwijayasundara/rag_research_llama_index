from dotenv import load_dotenv
import warnings
import nest_asyncio
import logging
import sys
import pandas as pd
from llama_index.core.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    RetrieverEvaluator,
    generate_question_context_pairs,
)
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Response,
)
from llama_index.llms.openai import OpenAI

nest_asyncio.apply()

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
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = SimpleDirectoryReader(
    input_files=["../docs/essay/paul_graham_essay.txt"]
).load_data()

# Generate Question
gpt4o = OpenAI(model="gpt-4o-mini", temperature=0.1)

dataset_generator = DatasetGenerator.from_documents(
    documents,
    llm=gpt4o,
    show_progress=True,
)

eval_dataset = dataset_generator.generate_dataset_from_nodes(num=2)

eval_queries = list(eval_dataset.queries.values())

eval_query = "How did the author describe their early attempts at writing short stories?"

# Fix GPT-4o-mini LLM for generating response
gpt4o_mini = OpenAI(temperature=0, model="gpt-4o-mini")

# Fix GPT-4o LLM for evaluation
gpt4o = OpenAI(temperature=0, model="gpt-4o")

# create vector index
vector_index = VectorStoreIndex.from_documents(documents, llm=gpt4o_mini)

# Query engine to generate response
query_engine = vector_index.as_query_engine()
retriever = vector_index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve(eval_query)

# print the nodes
for node in nodes:
    print(node)

# Faithfullness Evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=gpt4o)
response_vector = query_engine.query(eval_query)
eval_result = faithfulness_evaluator.evaluate_response(
    response=response_vector
)
print("Faithfullness evaluator passing? ", eval_result.passing)
print(eval_result)

# Relevency Evaluation
relevancy_evaluator = RelevancyEvaluator(llm=gpt4o)
response_vector = query_engine.query(eval_query)
eval_result = relevancy_evaluator.evaluate_response(
    query=eval_query, response=response_vector
)
print("Relevancy evaluator passing? ", eval_result.passing)
print(eval_result)

# Correctness Evaluator
correctness_evaluator = CorrectnessEvaluator(llm=gpt4o)

query = "Can you explain the theory of relativity proposed by Albert Einstein in detail?"

reference = """
Certainly! Albert Einstein's theory of relativity consists of two main components: special relativity and general 
relativity. Special relativity, published in 1905, introduced the concept that the laws of physics are the same for 
all non-accelerating observers and that the speed of light in a vacuum is a constant, regardless of the motion of the 
source or observer. It also gave rise to the famous equation E=mc², which relates energy (E) and mass (m).
General relativity, published in 1915, extended these ideas to include the effects of gravity. According to general 
relativity, gravity is not a force between masses, as described by Newton's theory of gravity, but rather the result 
of the warping of space and time by mass and energy. Massive objects, such as planets and stars, cause a curvature in 
spacetime, and smaller objects follow curved paths in response to this curvature. This concept is often illustrated 
using the analogy of a heavy ball placed on a rubber sheet, causing it to create a depression that other objects 
(representing smaller masses) naturally move towards.
In essence, general relativity provided a new understanding of gravity, explaining phenomena like the bending of light 
by gravity (gravitational lensing) and the precession of the orbit of Mercury. It has been confirmed through numerous 
experiments and observations and has become a fundamental theory in modern physics.
"""
response = """
Certainly! Albert Einstein's theory of relativity consists of two main components: special relativity and general 
relativity. Special relativity, published in 1905, introduced the concept that the laws of physics are the same for 
all non-accelerating observers and that the speed of light in a vacuum is a constant, regardless of the motion of the 
source or observer. It also gave rise to the famous equation E=mc², which relates energy (E) and mass (m).
However, general relativity, published in 1915, extended these ideas to include the effects of magnetism. According 
to general relativity, gravity is not a force between masses but rather the result of the warping of space and time by 
magnetic fields generated by massive objects. Massive objects, such as planets and stars, create magnetic fields that 
cause a curvature in spacetime, and smaller objects follow curved paths in response to this magnetic curvature. This 
concept is often illustrated using the analogy of a heavy ball placed on a rubber sheet with magnets underneath, 
causing it to create a depression that other objects (representing smaller masses) naturally move towards due to 
magnetic attraction.
"""

correctness_result = correctness_evaluator.evaluate(
    query=query,
    response=response,
    reference=reference,
)

print("correctness score :", correctness_result.score)
print("correctness passing? ", correctness_result.passing)
print("correctness feedback: ", correctness_result.feedback)

# Retrieval Evaluation
documents = SimpleDirectoryReader(
    input_files=["../docs/essay/paul_graham_essay.txt"]
).load_data()

from llama_index.core.text_splitter import SentenceSplitter

# create parser and parse document into nodes
parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
nodes = parser(documents)

vector_index = VectorStoreIndex(nodes)

retriever = vector_index.as_retriever(similarity_top_k=2)

retrieved_nodes = retriever.retrieve(eval_query)

from llama_index.core.response.notebook_utils import display_source_node

for node in retrieved_nodes:
    display_source_node(node, source_length=2000)

qa_dataset = generate_question_context_pairs(
    nodes, llm=gpt4o_mini, num_questions_per_chunk=2
)

queries = qa_dataset.queries.values()
print(list(queries)[5])

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

# try it out on a sample query
sample_id, sample_query = list(qa_dataset.queries.items())[0]
sample_expected = qa_dataset.relevant_docs[sample_id]

eval_result = retriever_evaluator.evaluate(sample_query, sample_expected)
print(eval_result)

async def evaluate_dataset(qa_dataset):
    """Evaluate a dataset asynchronously."""
    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
    return eval_results

eval_results = evaluate_dataset(qa_dataset)

def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}
    )

    return metric_df

print(eval_results)






