import os
from opik import Opik
from opik.evaluation import evaluate

# os.environ["OPENAI_API_KEY"] = "OpenAI API key goes here"

os.environ["OPIK_API_KEY"] = "mA6aWC00TFxabXeI91VEGQYmu" 
os.environ["OPIK_WORKSPACE"] = "rodiwaa"

from opik.evaluation.metrics import (Hallucination, AnswerRelevance, ContextRecall, ContextPrecision)
  
client = Opik()
dataset = client.get_dataset(name="Website Bot Queries")

def evaluation_task(dataset_item):
    # your LLM application is called here

    result = {
        "input": "placeholder string",
        "output": "placeholder string",
        "context": ["placeholder string"]
    }

    return result

metrics = [Hallucination(), AnswerRelevance(), ContextRecall(), ContextPrecision()]

eval_results = evaluate(
  experiment_name="site_bot_queries_evaluation",
  dataset=dataset,
  task=evaluation_task,
  scoring_metrics=metrics
)
