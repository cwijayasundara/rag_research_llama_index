from dotenv import load_dotenv
import warnings
import nest_asyncio
import pandas as pd
import phoenix as px
from llama_index.core import (
    set_global_handler,
)

from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from phoenix.session.evaluation import (
    get_qa_with_reference,
    get_retrieved_documents,
)
from phoenix.trace import DocumentEvaluations, SpanEvaluations
from tqdm import tqdm
from llama_index.core import SimpleDirectoryReader

nest_asyncio.apply()
pd.set_option("display.max_colwidth", 1000)

warnings.filterwarnings('ignore')
_ = load_dotenv()

session = px.launch_app()

documents = SimpleDirectoryReader(
    input_files=["../docs/essay/paul_graham_essay.txt"]
).load_data()

set_global_handler("arize_phoenix")

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
embed_model = OpenAIEmbedding()

Settings.llm = llm
Settings.embed_model = embed_model

from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=5)

queries = [
    "what did paul graham do growing up?",
    "why did paul graham start YC?",
]

for query in tqdm(queries):
    query_engine.query(query)

print(query_engine.query("Who is Paul Graham?"))

queries_df = get_qa_with_reference(px.Client())
retrieved_documents_df = get_retrieved_documents(px.Client())

eval_model = OpenAIModel(
    model="gpt-4o",
)
hallucination_evaluator = HallucinationEvaluator(eval_model)
qa_correctness_evaluator = QAEvaluator(eval_model)
relevance_evaluator = RelevanceEvaluator(eval_model)

hallucination_eval_df, qa_correctness_eval_df = run_evals(
    dataframe=queries_df,
    evaluators=[hallucination_evaluator, qa_correctness_evaluator],
    provide_explanation=True,
)
relevance_eval_df = run_evals(
    dataframe=retrieved_documents_df,
    evaluators=[relevance_evaluator],
    provide_explanation=True,
)[0]

px.Client().log_evaluations(
    SpanEvaluations(
        eval_name="Hallucination", dataframe=hallucination_eval_df
    ),
    SpanEvaluations(
        eval_name="QA Correctness", dataframe=qa_correctness_eval_df
    ),
    DocumentEvaluations(eval_name="Relevance", dataframe=relevance_eval_df),
)

