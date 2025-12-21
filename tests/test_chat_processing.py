import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
import backend.core.bootstrap as bootstrap
from backend.core.chat_processing import process_chat_message_streaming


@pytest.mark.asyncio
async def test_streaming_returns_correct_structure():
    dummy_chat_client = AsyncMock()
    response_mock = MagicMock()
    response_mock.content = "Testantwort vom Modell"
    dummy_chat_client.ainvoke.return_value = response_mock

    dummy_vectorstore = MagicMock()
    dummy_vectorstore.similarity_search.return_value = []

    dummy_memory = MagicMock()
    dummy_embeddings = MagicMock()
    dummy_embeddings.embed_query.return_value = [0.1] * 768

    gen = process_chat_message_streaming(
        "Was weißt du über Lexi?",
        chat_client=dummy_chat_client,
        vectorstore=dummy_vectorstore,
        memory=dummy_memory,
        embeddings=dummy_embeddings
    )

    first = await anext(gen)
    assert isinstance(first, dict)
    assert "chunk" in first
    assert isinstance(first["chunk"], str)
    assert "final_chunk" in first
    assert first["final_chunk"] is False


@pytest.mark.asyncio
async def test_streaming_short_input():
    dummy_chat_client = AsyncMock()
    response_mock = MagicMock()
    response_mock.content = "ok"
    dummy_chat_client.ainvoke.return_value = response_mock

    dummy_vectorstore = MagicMock()
    dummy_vectorstore.similarity_search.return_value = []

    dummy_memory = MagicMock()
    dummy_embeddings = MagicMock()
    dummy_embeddings.embed_query.return_value = [0.1] * 768

    gen = process_chat_message_streaming(
        "ok",
        chat_client=dummy_chat_client,
        vectorstore=dummy_vectorstore,
        memory=dummy_memory,
        embeddings=dummy_embeddings
    )

    first = await anext(gen)
    assert isinstance(first['chunk'], str)


@pytest.mark.asyncio
async def test_streaming_multiple_chunks():
    long_text = "Lexi ist eine KI, die sehr viele Dinge kann. Sie ist charmant, witzig und hilft dir immer gern weiter."

    dummy_chat_client = AsyncMock()
    response_mock = MagicMock()
    response_mock.content = long_text
    dummy_chat_client.ainvoke.return_value = response_mock

    dummy_vectorstore = MagicMock()
    dummy_vectorstore.similarity_search.return_value = []

    dummy_memory = MagicMock()
    dummy_embeddings = MagicMock()
    dummy_embeddings.embed_query.return_value = [0.1] * 768

    gen = process_chat_message_streaming(
        long_text,
        chat_client=dummy_chat_client,
        vectorstore=dummy_vectorstore,
        memory=dummy_memory,
        embeddings=dummy_embeddings
    )

    chunks = [chunk async for chunk in gen]
    assert len(chunks) > 1


@pytest.mark.asyncio
async def test_streaming_with_fallback_init(monkeypatch):
    # Set up all mocks first
    dummy_chat_client = AsyncMock()
    dummy_vectorstore = MagicMock()
    dummy_memory = MagicMock()
    dummy_embeddings = MagicMock()
    dummy_embeddings.embed_query.return_value = [0.1] * 768

    response_mock = MagicMock()
    response_mock.content = "Testantwort vom Modell"
    dummy_chat_client.ainvoke.return_value = response_mock

    # Mock the initialize_components function BEFORE calling process_chat_message_streaming
    def mock_initialize():
        return (dummy_embeddings, dummy_vectorstore, dummy_memory, dummy_chat_client, None)

    monkeypatch.setattr("backend.core.bootstrap.initialize_components", mock_initialize)

    # Now call the function
    gen = process_chat_message_streaming("Was ist deine Lieblingsfarbe?")
    first = await anext(gen)

    # Assertions
    assert isinstance(first, dict)
    assert "chunk" in first
    assert isinstance(first["chunk"], str)
    assert "final_chunk" in first
    assert first["final_chunk"] is False

