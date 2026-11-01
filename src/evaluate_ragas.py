import json
import re
import math
import os
from collections import Counter

# === 1. RAG: загрузка документов ===
def load_docs(docs_dir="data/docs"):
    docs = []
    for filename in os.listdir(docs_dir):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(docs_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, p in enumerate(paragraphs):
            docs.append({
                "id": f"{filename}::{i}",
                "source": filename,
                "text": p
            })
    return docs

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def build_index(chunks):
    doc_freq = Counter()
    for c in chunks:
        for t in set(tokenize(c["text"])):
            doc_freq[t] += 1
    return doc_freq

def score_chunk(query_tokens, chunk, doc_freq, total_docs):
    tokens = tokenize(chunk["text"])
    tf = Counter(tokens)
    s = 0.0
    for qt in query_tokens:
        if qt in tf:
            idf = math.log((total_docs + 1) / (doc_freq.get(qt, 0) + 1)) + 1
            s += tf[qt] * idf
    return s

def retrieve(query, chunks, doc_freq, top_k=3):
    q_tokens = tokenize(query)
    scored = [(score_chunk(q_tokens, c, doc_freq, len(chunks)), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def generate_answer(query, top_chunks):
    if not top_chunks or top_chunks[0][0] == 0:
        return "Не найдено."
    return f"На основе: {top_chunks[0][1]['source']}. {top_chunks[0][1]['text'][:200]}..."

# === 2. Метрики (упрощённые, вдохновлённые RAGAS) ===

def context_precision(retrieved, question):
    """Доля релевантных чанков в выдаче."""
    q_tokens = set(tokenize(question))
    relevant = 0
    for _, c in retrieved:
        c_tokens = set(tokenize(c["text"]))
        if q_tokens & c_tokens:
            relevant += 1
    return relevant / len(retrieved) if retrieved else 0.0

def context_recall(retrieved, ground_truth):
    """Доля найденных ключевых слов эталона."""
    gt_tokens = set(tokenize(ground_truth))
    if not gt_tokens:
        return 0.0
    found = set()
    for _, c in retrieved:
        found |= (gt_tokens & set(tokenize(c["text"])))
    return len(found) / len(gt_tokens)

def faithfulness(answer, retrieved):
    """Насколько ответ опирается на контекст."""
    ans_tokens = set(tokenize(answer))
    if not ans_tokens:
        return 0.0
    ctx_tokens = set()
    for _, c in retrieved:
        ctx_tokens |= set(tokenize(c["text"]))
    return len(ans_tokens & ctx_tokens) / len(ans_tokens)

def answer_relevancy(answer, question):
    """Насколько ответ релевантен вопросу."""
    ans_tokens = set(tokenize(answer))
    q_tokens = set(tokenize(question))
    if not q_tokens or not ans_tokens:
        return 0.0
    return len(ans_tokens & q_tokens) / len(q_tokens)

# === 3. Основной запуск ===
if __name__ == "__main__":
    chunks = load_docs()
    print(f"Загружено чанков: {len(chunks)}")
    doc_freq = build_index(chunks)

    with open("data/questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    totals = {"context_precision": 0.0, "context_recall": 0.0, "faithfulness": 0.0, "answer_relevancy": 0.0}

    for q in questions:
        retrieved = retrieve(q["question"], chunks, doc_freq, top_k=3)
        answer = generate_answer(q["question"], retrieved)

        cp = context_precision(retrieved, q["question"])
        cr = context_recall(retrieved, q["ground_truth"])
        f = faithfulness(answer, retrieved)
        ar = answer_relevancy(answer, q["question"])

        totals["context_precision"] += cp
        totals["context_recall"] += cr
        totals["faithfulness"] += f
        totals["answer_relevancy"] += ar

        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "ground_truth": q["ground_truth"],
            "context_precision": round(cp, 3),
            "context_recall": round(cr, 3),
            "faithfulness": round(f, 3),
            "answer_relevancy": round(ar, 3)
        })

        print(f"\n[{q['id']}] {q['question']}")
        print(f"  context_precision: {cp:.3f}")
        print(f"  context_recall:    {cr:.3f}")
        print(f"  faithfulness:      {f:.3f}")
        print(f"  answer_relevancy:  {ar:.3f}")

    n = len(questions)
    summary = {k: round(v / n, 3) for k, v in totals.items()}

    os.makedirs("results", exist_ok=True)
    with open("results/ragas_scores.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("RAGAS SUMMARY (упрощённые метрики)")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}")
    print("\nРезультаты сохранены в results/ragas_scores.json")
