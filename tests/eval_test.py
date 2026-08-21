"""Ragas quality gate; runs from ./eval-env against the running API."""
import json
import os

import pytest
import requests

from src.utils.text_sanitizer import REFUSAL_MARKERS, strip_citation_headers

try:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    RAGAS_INSTALLED = True
except ImportError:
    RAGAS_INSTALLED = False

API_URL = os.environ.get("VRAG_API_URL", "http://localhost:8000/api/query")
GOLDEN_PATH = "data/eval/golden_dataset.json"
SCORES_PATH = "data/eval/last_run_scores.csv"

# Lowered from 0.90: Ragas scores per claim, which penalises short answers.
FAITHFULNESS_THRESHOLD = 0.85
RELEVANCY_THRESHOLD = 0.85
REFUSAL_THRESHOLD = 1.00

# A local judge needs room; timed-out judgements return NaN and skew the mean.
JUDGE_TIMEOUT_SECONDS = 900
JUDGE_WORKERS = 4
MIN_JUDGED_FRACTION = 0.90

# Refusals assert nothing, so they are excluded here and coverage is asserted instead.
MIN_ANSWERED_FRACTION = 0.90

JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "llama3.1:8b")
JUDGE_EMBEDDINGS = os.environ.get("RAGAS_JUDGE_EMBEDDINGS", "qllama/bge-small-en-v1.5")

# The model phrases refusals several ways; all must count as a refusal.
REFUSALS = REFUSAL_MARKERS


def load_golden():
    if not os.path.exists(GOLDEN_PATH) or os.path.getsize(GOLDEN_PATH) == 0:
        pytest.skip(f"{GOLDEN_PATH} is missing. Run scripts/build_golden_dataset.py first.")
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def ask(question):
    response = requests.post(API_URL, json={"query": question}, timeout=300)
    response.raise_for_status()
    return response.json()


def is_refusal(answer):
    lowered = answer.lower()
    return any(phrase in lowered for phrase in REFUSALS)


@pytest.mark.ragas_ci
def test_out_of_scope_queries_are_refused():
    """Every adversarial question must be refused; no judge model involved."""
    rows = [r for r in load_golden() if not r["in_scope"]]
    assert rows, "The golden dataset has no out-of-scope questions."

    leaked = [r["question"] for r in rows if not is_refusal(ask(r["question"])["answer"])]

    refusal_rate = 1 - (len(leaked) / len(rows))
    print(f"\nRefusal rate: {refusal_rate:.2%} over {len(rows)} adversarial queries")
    for question in leaked:
        print(f"  LEAKED: {question}")

    assert refusal_rate >= REFUSAL_THRESHOLD, \
        f"{len(leaked)} out-of-scope questions produced an answer instead of a refusal."


@pytest.mark.ragas_ci
def test_pipeline_quality_gate():
    """Scores faithfulness and answer relevancy with a local Ollama judge."""
    if not RAGAS_INSTALLED:
        pytest.skip("Ragas is not installed. Use ./eval-env for this test.")

    rows = [r for r in load_golden() if r["in_scope"]]
    assert rows, "The golden dataset has no in-scope questions."

    samples = []
    for row in rows:
        result = ask(row["question"])
        contexts = result.get("retrieved_context") or []
        if not contexts or is_refusal(result["answer"]):
            # Gated out or refused: no claims were made, so there is nothing to judge.
            continue
        samples.append({
            "user_input": row["question"],
            "retrieved_contexts": contexts,
            # Citation markers are provenance, not claims.
            "response": strip_citation_headers(result["answer"]).strip(),
            "reference": row["ground_truth"],
        })

    answered_rate = len(samples) / len(rows)
    print(f"\nAnswered {len(samples)}/{len(rows)} in-scope questions ({answered_rate:.0%})")
    assert answered_rate >= MIN_ANSWERED_FRACTION, \
        f"Only {answered_rate:.0%} of in-scope questions were answered; the pipeline is refusing work it should do."

    judge = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=JUDGE_EMBEDDINGS))

    result = evaluate(
        EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(timeout=JUDGE_TIMEOUT_SECONDS, max_workers=JUDGE_WORKERS),
    )

    frame = result.to_pandas()
    # Persist per-sample scores so a failing aggregate can be diagnosed.
    frame.to_csv(SCORES_PATH, index=False)
    print(f"\nPer-sample scores written to {SCORES_PATH}")

    faithfulness_scores = frame["faithfulness"].dropna()
    relevancy_scores = frame["answer_relevancy"].dropna()

    print("\n--- RAGAS EVAL SCORES ---")
    for name, scored in (("faithfulness", faithfulness_scores), ("answer_relevancy", relevancy_scores)):
        judged = len(scored) / len(frame)
        print(f"{name:<17} {scored.mean():.3f}  (judged {len(scored)}/{len(frame)})")
        assert judged >= MIN_JUDGED_FRACTION, \
            f"The judge only scored {judged:.0%} of samples for {name}; the run is inconclusive, not a failure."

    score_faithfulness = faithfulness_scores.mean()
    score_relevancy = relevancy_scores.mean()

    assert score_faithfulness >= FAITHFULNESS_THRESHOLD, \
        f"Faithfulness score {score_faithfulness:.3f} failed the strict zero-hallucination threshold of {FAITHFULNESS_THRESHOLD}."

    assert score_relevancy >= RELEVANCY_THRESHOLD, \
        f"Relevancy score {score_relevancy:.3f} failed threshold of {RELEVANCY_THRESHOLD}."
