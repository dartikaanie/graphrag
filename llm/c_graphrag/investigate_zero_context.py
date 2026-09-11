"""
investigate_zero_context.py
=====================================
Script diagnostik SEKALI-PAKAI untuk menyelidiki kenapa pertanyaan tertentu
di Kondisi C menghasilkan context=0 (hybrid retrieval gagal total), TANPA
perlu re-run seluruh pipeline n=100 dan TANPA memanggil LLM sama sekali
(jadi tidak ada biaya API).

Cara pakai (dari folder c_graphrag/, sesudah c_graphrag.py di-update):
    python investigate_zero_context.py --question-ids 241991 52175918

Yang di-print:
  1. Anchor mana yang dipilih FAISS utk tiap pertanyaan
  2. Apakah tiap anchor punya edge HAS_ACCEPTED_ANSWER/HAS_ANSWER (1-hop)
  3. Apakah tiap anchor punya edge IS_RELATED_TO/TAG_COOCCUR/EMBED_SIM (2-hop)
  4. Semua ini dijalankan LANGSUNG via Cypher (tanpa exclusion apapun) utk
     mengecek apakah node-nya BENERAN terisolasi di KG, atau exclusion
     leakage-prevention yang (secara tidak sengaja) menghabiskan semua
     kandidat.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from dotenv import load_dotenv
load_dotenv()

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c_graphrag import (
    connect_neo4j, load_faiss_cache, embed_query, anchor_via_vector_search,
    traverse_graph, semantic_expansion, fuse_and_rank,
)

log = print


def get_question_texts(questions_parquet: str, answers_parquet: str, question_ids: list):
    con = duckdb.connect()
    ids_str = ",".join(str(int(i)) for i in question_ids)
    query = f"""
        SELECT Id, Title, Body, Tags, AcceptedAnswerId
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{questions_parquet}')
            WHERE Id IN ({ids_str})
        )
        WHERE rn = 1
    """
    df = con.execute(query).df()
    return df


def get_all_answer_ids(answers_parquet: str, question_ids: list) -> dict:
    con = duckdb.connect()
    ids_str = ",".join(str(int(i)) for i in question_ids)
    query = f"""
        SELECT ParentId AS question_id, Id AS answer_id
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{answers_parquet}')
            WHERE ParentId IN ({ids_str})
        )
        WHERE rn = 1
    """
    df = con.execute(query).df()
    return df.groupby("question_id")["answer_id"].apply(list).to_dict()


def raw_edge_check(driver, database, anchor_id: int):
    """Cek Cypher LANGSUNG tanpa exclusion apapun -- utk tahu apakah node
    ini beneran terisolasi di KG, atau exclusion yg menghabiskan semuanya."""
    with driver.session(database=database) as session:
        exists = session.run(
            "MATCH (q:Question {id: $id}) RETURN count(q) AS c", id=anchor_id
        ).single()["c"]

        n_direct_answers = session.run(
            """MATCH (q:Question {id: $id})-[:HAS_ACCEPTED_ANSWER|HAS_ANSWER]->(a:Answer)
               RETURN count(a) AS c""", id=anchor_id
        ).single()["c"]

        n_related = session.run(
            """MATCH (q:Question {id: $id})-[:IS_RELATED_TO|TAG_COOCCUR|EMBED_SIM]->(q2:Question)
               RETURN count(DISTINCT q2) AS c""", id=anchor_id
        ).single()["c"]

        n_related_with_answer = session.run(
            """MATCH (q:Question {id: $id})-[:IS_RELATED_TO|TAG_COOCCUR|EMBED_SIM]->(q2:Question)
                     -[:HAS_ACCEPTED_ANSWER|HAS_ANSWER]->(a:Answer)
               RETURN count(DISTINCT a) AS c""", id=anchor_id
        ).single()["c"]

    return {
        "node_exists_in_kg": exists > 0,
        "n_direct_answers_1hop": n_direct_answers,
        "n_related_questions_2hop": n_related,
        "n_related_with_answer_2hop": n_related_with_answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-ids", type=int, nargs="+", required=True)
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    parser.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"))
    parser.add_argument("--n-anchor", type=int, default=int(os.getenv("N_ANCHOR", 3)))
    parser.add_argument("--n-semantic-expansion", type=int, default=int(os.getenv("N_SEMANTIC_EXPANSION", 3)))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", 5)))
    parser.add_argument("--token-chunk-limit", type=int, default=int(os.getenv("TOKEN_CHUNK_LIMIT", 400)))
    parser.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--kg-workspace-dir", default=os.getenv("KG_WORKSPACE_DIR",
                         "../../01_data_cleaning/_kg_workspace"))
    args = parser.parse_args()

    if not args.questions_parquet or not args.answers_parquet:
        print("[ERROR] QUESTIONS_PARQUET / ANSWERS_PARQUET tidak ditemukan di .env")
        sys.exit(1)

    print(f"[setup] Menghubungkan Neo4j & memuat FAISS cache...")
    driver, database = connect_neo4j(log)
    faiss_index, faiss_ids, faiss_embeddings, id_to_row = load_faiss_cache(
        Path(args.kg_workspace_dir), log)

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.embed_model, device="cpu")

    q_df = get_question_texts(args.questions_parquet, args.answers_parquet, args.question_ids)
    all_answer_ids_map = get_all_answer_ids(args.answers_parquet, args.question_ids)

    for _, row in q_df.iterrows():
        qid = int(row["Id"])
        print("\n" + "=" * 78)
        print(f"INVESTIGASI Id={qid}")
        print(f"Title: {row['Title']}")
        print("=" * 78)

        exclude_question_ids = {qid}
        exclude_answer_ids = set(all_answer_ids_map.get(qid, []))
        if not (row["AcceptedAnswerId"] is None or (isinstance(row["AcceptedAnswerId"], float) and np.isnan(row["AcceptedAnswerId"]))):
            exclude_answer_ids.add(int(row["AcceptedAnswerId"]))

        if qid in id_to_row:
            query_emb = faiss_embeddings[id_to_row[qid]:id_to_row[qid] + 1]
            query_emb = np.ascontiguousarray(query_emb)
            print(f"[info] embedding diambil dari FAISS cache (qid ada di index)")
        else:
            query_text = f"{row['Title']} {str(row['Body'])[:1000]}"
            query_emb = embed_query(query_text, embed_model)
            print(f"[info] qid TIDAK ADA di FAISS cache -- re-embed dari teks "
                  f"(kemungkinan pertanyaan ini tidak termasuk populasi saat KG dibangun)")

        # --- a. Entity Anchoring ---
        anchors = anchor_via_vector_search(
            query_emb, faiss_index, faiss_ids, args.n_anchor, exclude_question_ids)
        anchor_ids = [a for a, _ in anchors]
        print(f"\n[anchors] {len(anchor_ids)} anchor terpilih: {anchors}")

        print(f"\n[raw-edge-check] Cek Cypher LANGSUNG (tanpa exclusion apapun) utk tiap anchor:")
        for aid in anchor_ids:
            info = raw_edge_check(driver, database, aid)
            print(f"  anchor={aid}: {info}")
            if not info["node_exists_in_kg"]:
                print(f"    >>> PENYEBAB: node {aid} TIDAK ADA di Neo4j KG sama sekali "
                      f"(ada di FAISS index tapi tidak pernah di-ingest ke graf -- "
                      f"cek pipeline 7_build_knowledge_graph.py / populasi KG vs index)")
            elif info["n_direct_answers_1hop"] == 0 and info["n_related_with_answer_2hop"] == 0:
                print(f"    >>> PENYEBAB: node {aid} ADA di KG tapi BENAR-BENAR TERISOLASI "
                      f"-- tidak punya jawaban sendiri DAN tidak punya related-question "
                      f"manapun yang berjawaban (baik lewat IS_RELATED_TO, TAG_COOCCUR, "
                      f"maupun EMBED_SIM). Ini kandidat pertanyaan yang secara struktural "
                      f"'yatim' di KG -- densifikasi (script 10_/11_) mungkin belum "
                      f"menjangkau node ini.")
            elif info["n_direct_answers_1hop"] == 0:
                print(f"    >>> node {aid} tidak punya jawaban sendiri, tapi PUNYA "
                      f"{info['n_related_with_answer_2hop']} kandidat via 2-hop -- "
                      f"traverse_graph() SEHARUSNYA masih dapat kandidat dari sini "
                      f"(exclusion mungkin yang menghabiskannya, lihat traversal di bawah)")

        # --- Jalankan pipeline ASLI (dengan exclusion) utk lihat titik gagalnya ---
        print(f"\n[pipeline-asli] Menjalankan traverse_graph() + semantic_expansion() + "
              f"fuse_and_rank() PERSIS seperti run sebenarnya (dgn exclusion leakage):")
        graph_candidates = []
        if anchor_ids:
            graph_candidates = traverse_graph(
                driver, database, anchor_ids, exclude_question_ids, exclude_answer_ids)

        seen_qids = exclude_question_ids | set(anchor_ids)
        expansion_candidates = semantic_expansion(
            driver, database, graph_candidates, faiss_index, faiss_ids, embed_model,
            args.n_semantic_expansion, seen_qids, exclude_question_ids, exclude_answer_ids)

        retrieved = fuse_and_rank(graph_candidates, expansion_candidates,
                                   args.top_k, args.token_chunk_limit)

        print(f"\n[HASIL AKHIR] Id={qid}: {len(retrieved)} context item "
              f"(graph_candidates={len(graph_candidates)}, "
              f"expansion_candidates={len(expansion_candidates)})")

    driver.close()


if __name__ == "__main__":
    sys.exit(main())