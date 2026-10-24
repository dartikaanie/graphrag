"""Per-item provenance (llm/common/item_provenance.py) must never reach the LLM:
the prompt for items that carry the FULL provenance equals, byte for byte, the
prompt for the same items with only question_id + chunk_text, and the context
block is exactly `[SO-<id>] <text>` per item. No LLM calls."""

import json

import pytest

from llm.common import item_provenance as ip
from llm.prompts import _format_context_block, build_graphrag_messages, build_lightrag_messages, build_rag_messages

# values that cannot occur in a prompt by accident
SENTINEL_NODE_ID = 987654321
SENTINEL_ALT_NODE_ID = 123456789


def _full_item(question_id: int, text: str) -> dict:
    path = ip.two_hop_path(SENTINEL_NODE_ID, question_id, 555, "TAG_COOCCUR", "HAS_ANSWER")
    alt = {"path": ip.one_hop_path(SENTINEL_ALT_NODE_ID, 555, "HAS_ANSWER"), "source_stage": "graph_traversal",
           "hop": 1, "rel_type": "HAS_ANSWER", "combined_score": 0.31}
    return {"question_id": question_id, "answer_id": 555, "chunk_text": text,
            "trust_weight": 0.5681, "combined_score": 0.6543, "hop": 2, "rel_type": "TAG_COOCCUR->HAS_ANSWER",
            "source_stage": "graph_traversal", "is_accepted": True, "sim": 0.4242, "sim_norm": 0.9, "trust": 0.5,
            "final_score": 0.7777, "alpha": 0.5, "relevance_score": 0.8123,
            **ip.provenance_fields(source_stage="graph_traversal", hop=2, path=path, rank=1, via_question_id=question_id,
                                   alt_paths=[alt])}


FORBIDDEN = ["hop", "rel_type", "path", "trust_weight", "source_stage", "anchor_id", "alt_paths", "via_question_id",
             "seed_similarity", "edge_type", "node_id", "node_type", "provenance", "combined_score", "final_score",
             "relevance_score", "TAG_COOCCUR", "HAS_ANSWER", "SEMANTIC_SIM", str(SENTINEL_NODE_ID),
             str(SENTINEL_ALT_NODE_ID), "0.5681", "0.6543", "0.8123", "0.7777", "0.4242"]

ITEMS = [_full_item(111, "first context item"), _full_item(222, "second context item")]
BARE = [{"question_id": i["question_id"], "chunk_text": i["chunk_text"]} for i in ITEMS]


def _builders():
    return {
        "B": lambda r, g: build_rag_messages("Title", "Body", "<python>", r, require_citation=True, require_grounding=g),
        "C": lambda r, g: build_graphrag_messages("Title", "Body", "<python>", r, require_grounding=g),
        "D": lambda r, g: build_lightrag_messages("Title", "Body", "<python>", r, require_grounding=g),
    }


def test_context_block_is_label_and_text_only():
    assert _format_context_block(ITEMS) == "[SO-111] first context item\n\n[SO-222] second context item"


@pytest.mark.parametrize("condition", ["B", "C", "D"])
@pytest.mark.parametrize("grounded", [True, False])
def test_prompt_with_full_provenance_equals_prompt_without_it(condition, grounded):
    build = _builders()[condition]
    assert build(ITEMS, grounded) == build(BARE, grounded)


@pytest.mark.parametrize("condition", ["B", "C", "D"])
def test_prompt_contains_no_provenance_field_or_value(condition):
    blob = json.dumps(_builders()[condition](ITEMS, True) + _builders()[condition](ITEMS, False))
    # the prompt's own instructions are English prose; only check words/values that appear nowhere in it
    # when provenance is absent -- i.e. anything present here but absent from the bare prompt is a leak
    bare = json.dumps(_builders()[condition](BARE, True) + _builders()[condition](BARE, False))
    leaked = [f for f in FORBIDDEN if f in blob and f not in bare]
    assert leaked == []
    assert blob == bare
