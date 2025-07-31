import os
import pytest
from dotenv import load_dotenv

from rag_src.Complete_RAG_Pipeline.CRAG import CRAG
from rag_src.llm.groq import GroqLLM
from rag_src.llm.gemini import GeminiLLM
from rag_src.doc_loader.universal_doc_loader import UniversalDocLoader
from rag_src.chunker import DefaultChunker
from rag_src.embedder import DefaultEmbedder
from rag_src.indexer import DefaultIndexer
from rag_src.retriever import DefaultRetriever
from rag_src.web_retriever import TavilyWebRetriever
from rag_src.query_transformer import LLMWebQueryTransformer, DefaultQueryTransformer
from rag_src.post_retrival_enricher import PostDefaultEnricher
from rag_src.doc_preprocessor import DefaultPreprocessor
from rag_src.evaluator import RelevanceEvaluator


@pytest.mark.skipif(
    not os.path.exists("tests/assests/sample1.pdf"),
    reason="PDF document missing for test",
)
def test_crag_with_groq_complete_coverage():
    """Test CRAG with GroqLLM for complete coverage without mocks"""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    assert api_key is not None, "GROQ_API_KEY is missing in .env"

    # Test 1: Initialize CRAG with all components
    crag = CRAG(
        llm=GroqLLM(api_key=api_key),
        docdir="tests/assests/sample1.pdf"
    )
    
    # Test constructor initialization
    assert crag.docdir == "tests/assests/sample1.pdf"
    assert isinstance(crag.llm, GroqLLM)
    assert isinstance(crag.embeddor, DefaultEmbedder)
    assert isinstance(crag.indexer, DefaultIndexer)
    assert isinstance(crag.query_transform, LLMWebQueryTransformer)
    assert isinstance(crag.doc_enricher, PostDefaultEnricher)
    # CRAG now uses UniversalDocLoader for better file format support
    assert isinstance(crag.doc_loader, UniversalDocLoader)
    assert isinstance(crag.preprocessor, DefaultPreprocessor)
    assert isinstance(crag.chunker, DefaultChunker)
    assert isinstance(crag.evaluator, RelevanceEvaluator)
    assert isinstance(crag.web_retriever, TavilyWebRetriever)
    assert isinstance(crag.retriever, DefaultRetriever)

    # Test 2: Run main query processing pipeline
    query = "What is Retrieval-Augmented Generation?"
    answer = crag.run(query)

    # Handle different LLM output formats
    if hasattr(answer, "text"):
        answer = answer.text

    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer.strip()) > 0, "Answer should not be empty"
    print(f"\n✅ CRAG Answer: {answer}")

    # Test 3: Test with custom components
    custom_chunker = DefaultChunker(chunk_size=256, chunk_overlap=25)
    custom_embedder = DefaultEmbedder()
    custom_indexer = DefaultIndexer(persist_path="test_crag_index")
    
    crag_custom = CRAG(
        llm=GroqLLM(api_key=api_key),
        embeddor=custom_embedder,
        indexer=custom_indexer,
        chunker=custom_chunker,
        docdir="tests/assests/sample1.pdf"
    )
    
    # Test custom initialization
    assert crag_custom.embeddor == custom_embedder
    assert crag_custom.indexer == custom_indexer
    assert crag_custom.chunker == custom_chunker
    
    # Test 4: Test different query types to exercise different code paths
    queries = [
        "Who is mentioned in this document?",
        "What are the key concepts discussed?",
        "How does RAG work?"
    ]
    
    for test_query in queries:
        result = crag.run(test_query)
        if hasattr(result, "text"):
            result = result.text
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        print(f"✅ Query: '{test_query}' -> Answer length: {len(result)}")


@pytest.mark.skipif(
    not os.path.exists("tests/assests/sample1.pdf"),
    reason="PDF document missing for test",
)
def test_crag_document_ingestion_methods():
    """Test CRAG document loading and ingestion methods"""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if api_key is None:
        pytest.skip("GROQ_API_KEY is missing in .env")

    # Create CRAG instance with custom indexer path to avoid conflicts
    test_indexer = DefaultIndexer(persist_path="test_ingestion_index")
    crag = CRAG(
        llm=GroqLLM(api_key=api_key),
        indexer=test_indexer,
        docdir="tests/assests/sample1.pdf"
    )

    # Test manual document ingestion
    test_documents = [
        "This is a test document about artificial intelligence.",
        "Machine learning is a subset of AI that focuses on algorithms.",
        "RAG combines retrieval and generation for better responses."
    ]
    
    # Test ingest_documents method
    crag.ingest_documents(test_documents)
    
    # Verify index was created
    index_path = getattr(crag.indexer, "persist_path", "default_index")
    index_file = os.path.join(index_path, "index.faiss")
    assert os.path.exists(index_file), "Index file should be created"

    # Test load_and_ingest_documents method by creating new instance
    # that will trigger the document loading pipeline
    os.remove(index_file) if os.path.exists(index_file) else None
    
    crag_reload = CRAG(
        llm=GroqLLM(api_key=api_key),
        indexer=DefaultIndexer(persist_path="test_reload_index"),
        docdir="tests/assests/sample1.pdf"
    )
    
    # The constructor should have called load_and_ingest_documents
    reload_index_path = getattr(crag_reload.indexer, "persist_path", "default_index")
    reload_index_file = os.path.join(reload_index_path, "index.faiss")
    assert os.path.exists(reload_index_file), "Index should be created during initialization"

    # Test query after manual ingestion
    query = "What is machine learning?"
    answer = crag_reload.run(query)
    if hasattr(answer, "text"):
        answer = answer.text
    
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
    print(f"✅ Post-ingestion query result: {answer}")


def test_crag_with_gemini():
    """Test CRAG with GeminiLLM to cover different LLM backends"""
    load_dotenv()
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if gemini_key is None:
        pytest.skip("GOOGLE_API_KEY is missing in .env")

    if not os.path.exists("tests/assests/sample1.pdf"):
        pytest.skip("PDF document missing for test")

    # Test with Gemini LLM
    crag_gemini = CRAG(
        llm=GeminiLLM(gemini_key),
        indexer=DefaultIndexer(persist_path="test_gemini_index"),
        docdir="tests/assests/sample1.pdf"
    )

    query = "Summarize the main points in this document"
    answer = crag_gemini.run(query)
    
    if hasattr(answer, "text"):
        answer = answer.text

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
    print(f"✅ Gemini CRAG Answer: {answer}")


def test_crag_error_conditions():
    """Test CRAG error handling and edge cases"""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if api_key is None:
        pytest.skip("GROQ_API_KEY is missing in .env")

    # Test with non-existent document path
    try:
        crag_error = CRAG(
            llm=GroqLLM(api_key=api_key),
            docdir="non_existent_file.pdf"
        )
        # Should still initialize but may have issues during document loading
        assert isinstance(crag_error, CRAG)
    except Exception as e:
        # This is acceptable - the constructor might fail with invalid paths
        print(f"Expected error with invalid docdir: {e}")

    # Test with empty query
    if os.path.exists("tests/assests/sample1.pdf"):
        crag = CRAG(
            llm=GroqLLM(api_key=api_key),
            indexer=DefaultIndexer(persist_path="test_error_index"),
            docdir="tests/assests/sample1.pdf"
        )
        
        # Test empty query
        empty_result = crag.run("")
        if hasattr(empty_result, "text"):
            empty_result = empty_result.text
        assert isinstance(empty_result, str)
        
        # Test very long query
        long_query = "What is " + "very " * 100 + "long query about the document?"
        long_result = crag.run(long_query)
        if hasattr(long_result, "text"):
            long_result = long_result.text
        assert isinstance(long_result, str)


def test_crag_component_coverage():
    """Test CRAG with explicit component parameters for full coverage"""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if api_key is None:
        pytest.skip("GROQ_API_KEY is missing in .env")

    if not os.path.exists("tests/assests/sample1.pdf"):
        pytest.skip("PDF document missing for test")

    # Test with all explicit components to ensure full coverage
    custom_components = CRAG(
        llm=GroqLLM(api_key=api_key),
        embeddor=DefaultEmbedder(),
        indexer=DefaultIndexer(persist_path="test_component_index"),
        retriever=None,  # Will use default
        web_retriever=TavilyWebRetriever(),
        evaluator=RelevanceEvaluator(llm=GroqLLM(api_key=api_key)),
        query_transform=LLMWebQueryTransformer(GroqLLM(api_key=api_key)),
        doc_enricher=PostDefaultEnricher(),
        doc_loader=UniversalDocLoader("tests/assests/sample1.pdf"),
        preprocessor=DefaultPreprocessor(),
        chunker=DefaultChunker(chunk_size=512, chunk_overlap=50),
        docdir="tests/assests/sample1.pdf"
    )

    # Test that all components are properly set
    assert isinstance(custom_components.llm, GroqLLM)
    assert isinstance(custom_components.embeddor, DefaultEmbedder)
    assert isinstance(custom_components.indexer, DefaultIndexer)
    assert isinstance(custom_components.web_retriever, TavilyWebRetriever)
    assert isinstance(custom_components.evaluator, RelevanceEvaluator)
    assert isinstance(custom_components.query_transform, LLMWebQueryTransformer)
    assert isinstance(custom_components.doc_enricher, PostDefaultEnricher)
    assert isinstance(custom_components.doc_loader, UniversalDocLoader)
    assert isinstance(custom_components.preprocessor, DefaultPreprocessor)
    assert isinstance(custom_components.chunker, DefaultChunker)

    # Test the run method with custom components
    query = "What technologies are discussed in this document?"
    answer = custom_components.run(query)
    
    if hasattr(answer, "text"):
        answer = answer.text

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
    print(f"✅ Custom components answer: {answer}")

    # Test to trigger web retrieval fallback by using irrelevant query
    irrelevant_query = "What is the weather like in Mars today?"
    web_answer = custom_components.run(irrelevant_query)
    
    if hasattr(web_answer, "text"):
        web_answer = web_answer.text
        
    assert isinstance(web_answer, str)
    print(f"✅ Web fallback answer: {len(web_answer)} characters")
