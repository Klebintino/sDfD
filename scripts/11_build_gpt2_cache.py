"""Build the GPT-2 input-embedding cache, using the same word set as Word2Vec.

Each word's vector is the mean of the input-embedding rows for the BPE tokens of
" " + word. Words Word2Vec could not resolve are left out. Needs the Word2Vec
cache from script 05, plus torch and transformers (downloads gpt2, ~500 MB).

Run:  python scripts/11_build_gpt2_cache.py
Writes outputs/emb_gpt2.npz
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import collections

import numpy as np

from sdfd import data, normalise, vocab as V

RAW = ROOT / "data" / "raw"
XLSX = RAW / "ucsd_fluency_for_snafu_20180518.xlsx"
CSV = RAW / "ucsd_fluency.csv"
OUT = ROOT / "outputs"
W2V_CACHE = OUT / "emb_word2vec_google_news_300.npz"


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not W2V_CACHE.exists():
        sys.exit(f"{W2V_CACHE.name} not found -- run 05_build_cache.py first. "
                 "The GPT-2 vocabulary is matched to it.")

    # ---- the word set and splits Word2Vec resolved ----
    w2v = V.CachedEmbedder(W2V_CACHE)
    animals = normalise.load_animal_list(RAW)
    sess = data.load_sessions(XLSX)
    sess = normalise.merge_into_sessions(sess, normalise.load_normalised(RAW, CSV))
    counts = collections.Counter()
    for ws in sess.unique_words:
        for w in ws:
            counts[w] += 1

    resolved = set(w2v.words)
    print(f"matching Word2Vec vocabulary: {len(resolved)} words resolved, "
          f"{len(counts) - len(resolved)} deliberately excluded")

    # reconstruct how each word was split, using the same rule
    plan = {}
    for w in resolved:
        method = str(w2v.methods.get(w, "direct"))
        if method == "compound":
            sp = V.split_modifier_animal(w, animals)
            plan[w] = list(sp) if sp else [w]
        else:
            plan[w] = [w]

    # ---- GPT-2 input embeddings ----
    print("loading GPT-2 (first run downloads ~500 MB) ...")
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2")
    emb_layer = model.get_input_embeddings()
    E = emb_layer.weight.detach().cpu().numpy()
    print(f"  input embedding matrix: {E.shape[0]:,} tokens x {E.shape[1]} dims")

    def word_vector(word: str) -> np.ndarray:
        # leading space = the form used in running text
        ids = tok.encode(" " + word)
        return E[ids].mean(axis=0)

    words, vecs, npieces = [], [], []
    for w, parts in plan.items():
        pv = [word_vector(p) for p in parts]
        v = np.mean(pv, axis=0)
        n = np.linalg.norm(v)
        if n == 0:
            continue
        words.append(w)
        vecs.append(v / n)
        npieces.append(len(tok.encode(" " + w)))

    path = OUT / "emb_gpt2.npz"
    np.savez_compressed(
        path,
        words=np.array(words),
        vectors=np.vstack(vecs).astype(np.float32),
        methods=np.array([str(w2v.methods.get(w, "direct")) for w in words]),
    )

    print(f"\n  cached {len(words)} words, {E.shape[1]} dims")
    print(f"  BPE pieces per word: mean {np.mean(npieces):.2f}, "
          f"max {max(npieces)}, single-token {sum(1 for n in npieces if n == 1)}")
    print(f"  identical vocabulary to Word2Vec cache: "
          f"{set(words) == resolved}")
    print(f"\nwrote {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    print("\nnext:")
    print("  python scripts/06_semantic_features.py gpt2")
    print("  python scripts/07_models.py A")
    print("  python scripts/08_ablation.py A")


if __name__ == "__main__":
    main()
