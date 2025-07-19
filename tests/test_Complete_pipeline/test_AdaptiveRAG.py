import os
import pytest
from dotenv import load_dotenv

from rag_src.Complete_RAG_Pipeline.AdaptiveRAG import AdaptiveRAG
from rag_src.doc_loader.universal_doc_loader import UniversalDocLoader
from rag_src.llm import GroqLLM


@pytest.mark.skipif(
    not os.path.exists(r"tests/assests/sample1.pdf"),
    reason="PDF document missing for test",
)
@pytest.mark.parametrize("query,expected_category", [
    ("What is the capital of France?", "Factual"),
    ("Analyze the economic impact of COVID-19", "Analytical"),
    ("What are different opinions on AI replacing jobs?", "Opinion"),
    ("Given my background in physics, how should I learn machine learning?", "Contextual"),
])
def test_adaptive_rag_response_for_each_strategy(query, expected_category):
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    assert api_key is not None, "GROQ_API_KEY is missing in .env"

    # Setup AdaptiveRAG pipeline
    docdir = r"tests/assests/sample1.pdf"
    rag_pipeline = AdaptiveRAG(
        llm=GroqLLM(api_key=api_key),
        docdir=docdir,
        doc_loader=UniversalDocLoader(docdir),
    )

    # Inject tracking for test observability (optional but useful)
    answer = rag_pipeline.run(query)
    actual_category = getattr(rag_pipeline, "last_category", None)

    # Accept either string or LLM text object
    if hasattr(answer, "text"):
        answer = answer.text

    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer.strip()) > 0, "Answer should not be empty"
    
    if actual_category:
        print(f"\nQuery classified as: {actual_category}")
        assert actual_category == expected_category, f"Expected: {expected_category}, Got: {actual_category}"

    print("\nAdaptiveRAG Answer:\n", answer)
