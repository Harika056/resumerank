import numpy as np
import pytest

from app.config import Settings
from app.embeddings import HashEmbedder, SentenceTransformerEmbedder, get_embedder, tokenize


def test_tokenize_keeps_cpp_and_drops_stopwords():
    assert tokenize("The C++ and C# developers") == ["c++", "c#", "developers"]


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder(64)
    a = e.embed(["python fastapi docker"])
    b = e.embed(["python fastapi docker"])
    assert np.array_equal(a, b)
    assert a.shape == (1, 64)
    assert a.dtype == np.float32
    assert np.linalg.norm(a[0]) == pytest.approx(1.0, abs=1e-5)


def test_hash_embedder_similar_text_scores_higher_than_unrelated():
    e = HashEmbedder()
    v = e.embed(
        ["python pytorch fastapi model serving", "python pytorch model serving api", "kubernetes terraform aws"]
    )
    assert v[0] @ v[1] > v[0] @ v[2]


def test_hash_embedder_empty_inputs():
    e = HashEmbedder(32)
    assert e.embed([]).shape == (0, 32)
    assert np.allclose(e.embed([""]), 0.0)  # no tokens -> zero vector, no crash


def test_hash_embedder_rejects_tiny_dim():
    with pytest.raises(ValueError):
        HashEmbedder(4)


class FakeModel:
    """Stands in for a SentenceTransformer: returns a vector per chunk."""

    def __init__(self):
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, chunks, normalize_embeddings=True, convert_to_numpy=True):
        self.calls.append(list(chunks))
        rows = []
        for c in chunks:
            v = np.array([len(c.split()), 1.0, 0.0, 0.0], dtype=np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows)


def test_sentence_transformer_embedder_chunks_long_text_and_normalises():
    model = FakeModel()
    e = SentenceTransformerEmbedder(chunk_words=5, model=model)
    text = " ".join(f"w{i}" for i in range(12))  # 12 words -> 3 chunks (5, 5, 2)
    out = e.embed([text, "short text"])
    assert [len(c.split()) for c in model.calls[0]] == [5, 5, 2, 2]
    assert out.shape == (2, 4)
    assert np.linalg.norm(out, axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)


def test_sentence_transformer_embedder_chunking_can_be_disabled():
    model = FakeModel()
    e = SentenceTransformerEmbedder(chunk_words=0, model=model)
    e.embed([" ".join(["x"] * 500)])
    assert len(model.calls[0]) == 1


def test_get_embedder_selects_backend_and_rejects_unknown():
    assert isinstance(get_embedder(Settings(embedder="hash")), HashEmbedder)
    with pytest.raises(ValueError):
        get_embedder(Settings(embedder="nope"))
