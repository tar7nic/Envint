import pytest
from app.ingestion.chunker import MalformedDocumentError, chunk_file, chunk_text, content_hash, load_corpus

DOC = "# POL-100: Test\nversion: 3\n## 1. Section\n1.1 First rule must hold.\n1.2 Second rule may apply.\n"


def test_chunk_text_happy_path():
    chunks = chunk_text(DOC)
    assert [c.chunk_id for c in chunks] == ["POL-100#1.1", "POL-100#1.2"]
    assert chunks[0].section == "1. Section" and chunks[0].doc_version == "3"


@pytest.mark.parametrize("bad", ["", "   \n", "1.1 no header at all", "# POL-9: x\njust prose, no clauses"])
def test_malformed_documents_rejected(bad):
    with pytest.raises(MalformedDocumentError):
        chunk_text(bad)


def test_content_hash_ignores_whitespace_but_not_content():
    assert content_hash("a  b\n c") == content_hash("a b c")
    assert content_hash("a b c") != content_hash("a b d")


def test_non_utf8_file_rejected(tmp_path):
    f = tmp_path / "bad.md"
    f.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(MalformedDocumentError):
        chunk_file(f)


def test_load_corpus_skips_bad_file_but_keeps_good(tmp_path):
    (tmp_path / "good.md").write_text(DOC, encoding="utf-8")
    (tmp_path / "bad.md").write_text("garbage", encoding="utf-8")
    chunks, failed = load_corpus(tmp_path)
    assert len(chunks) == 2 and failed[0][0] == "bad.md"
