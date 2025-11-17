from sentence_transformers import SentenceTransformer
import faiss, json, argparse, numpy as np

def semantic_search(query, model, index, chunks, top_k=5):
    q_embed = model.encode([query])
    distances, indices = index.search(np.array(q_embed).astype("float32"), top_k)
    return [
        {
            "rank": i + 1,
            "score": float(distances[0][i]),
            "chunk": chunks[idx]["text"],
            "source": chunks[idx]["source"]
        }
        for i, idx in enumerate(indices[0])
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    index = faiss.read_index("index.faiss")
    chunks = json.load(open("chunks.json"))

    results = semantic_search(args.query, model, index, chunks, args.top_k)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
