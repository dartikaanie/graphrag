"""Text representations shared by Conditions B and C (reports/b_corpus_parity.md).

- build_query_text(): the retrieval query text. Byte-identical to the
  expression Condition C (and D) has always used in its embed_query()
  fallback, f"{Title} {str(Body)[:1000]}", which is also the format
  01_data_cleaning/11_densify_embedding_similarity.py encoded every
  Question with (title + " " + body[:1000]). The ONE difference between the
  two: script 11 maps a NULL title/body to "" (`r["title"] or ""`), this
  function -- like C -- formats it as "None". No evaluation question has a
  NULL title or body (checked on the dev split), so the two agree wherever
  it matters; kept as C has it so C's behaviour does not change.
- answer_doc_text(): one answer as a retrieval document,
  f"Q: {question title}\nA: {answer body}" -- the text C builds per
  candidate in fuse_and_rank() and B indexes per answer.
- truncate_doc_text(): cut a document to its first `token_chunk_limit`
  cl100k_base tokens (moved here unchanged from c_graphrag._truncate_doc_text).
"""

QUERY_BODY_CHARS = 1000
TOKENIZER_NAME = "cl100k_base"


def build_query_text(title, body) -> str:
    return f"{title} {str(body)[:QUERY_BODY_CHARS]}"


def answer_doc_text(question_title, answer_body) -> str:
    return f"Q: {question_title}\nA: {answer_body}"


def truncate_doc_text(enc, doc_text: str, token_chunk_limit: int) -> str:
    token_ids = enc.encode(doc_text)
    if len(token_ids) > token_chunk_limit:
        return enc.decode(token_ids[:token_chunk_limit])
    return doc_text


def get_tokenizer():
    import tiktoken

    return tiktoken.get_encoding(TOKENIZER_NAME)
