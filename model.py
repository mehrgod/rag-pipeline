"""
RAG Pipeline

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_text_file
def load_text_file(path):
    # TODO: read a UTF-8 text file at `path` and return its contents as one string.
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    return content

# Step 2 - load_text_directory
import os

def load_text_directory(directory):
    # TODO: read every .txt file in `directory` and return their contents as a list of strings
    
    
    lst = []
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if full_path.endswith(".txt"):
            lst.append(full_path)

    lst.sort()
        
    out = []
    for l in lst:
        txt = load_text_file(l)
        out.append(txt)
    
    return out

# Step 3 - extract_text_from_html
from html.parser import HTMLParser


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if self.skip_depth == 0:
            self.text_parts.append(data)

    def get_text(self):
        # return "".join(part.strip() for part in self.text_parts if part.strip())
        return "".join(part for part in self.text_parts if part)


def extract_text_from_html(html):
    # TODO: strip HTML tags and return only the visible text content

    parser = TextExtractor()
    parser.feed(html)
    return parser.get_text()

# Step 4 - normalize_text
import unicodedata

def normalize_text(text):
    # TODO: NFKC-normalize the text and collapse runs of whitespace into single spaces.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")

    lst = text.split()

    out = " ".join(lst)
    
    return out

# Step 5 - make_document
def make_document(text, source, title):
    # TODO: wrap text with source and title metadata into a document dict.
    d = {"text": text,
            "source": source,
            "title": title}

    return d

# Step 6 - chunk_fixed_size
def chunk_fixed_size(text, chunk_size):
    # TODO: split text into consecutive non-overlapping chunks of length chunk_size
    chunks = []
    while len(text) > chunk_size:
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]

    if len(text) > 0:
        chunks.append(text)

    return chunks

# Step 7 - chunk_by_tokens
from transformers import AutoTokenizer

def chunk_by_tokens(text, tokenizer, max_tokens):
    # TODO: split text into chunks of at most max_tokens token ids using the tokenizer
    tokens = tokenizer.encode(text)
 
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunks.append(text[i:i+max_tokens])
    
    return chunks

# Step 8 - chunk_by_sentences
def chunk_by_sentences(text, max_chars):
    # TODO: split text on .!? boundaries and greedily pack whole sentences under max_chars.

    if not text or not text.strip():
        return []

    # Split into sentences
    sentences = []
    start = 0

    for i, ch in enumerate(text):
        if ch in ".!?":
            sentence = text[start:i + 1].strip()
            if sentence:
                sentences.append(sentence)
            start = i + 1

    # Handle trailing text without punctuation
    if start < len(text):
        sentence = text[start:].strip()
        if sentence:
            sentences.append(sentence)

    # Greedily pack sentences into chunks
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Long sentence gets its own chunk
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.append(sentence)
            continue

        if not current_chunk:
            current_chunk = sentence
        else:
            candidate = current_chunk + " " + sentence

            if len(candidate) <= max_chars:
                current_chunk = candidate
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Step 9 - chunk_with_overlap
def chunk_with_overlap(text, chunk_size, overlap):
    # TODO: return sliding-window chunks of length chunk_size sharing `overlap` chars
    chunks = []

    for i in range(0, len(text), chunk_size-overlap):
        chunks.append(text[i:i+chunk_size])

    return chunks

# Step 10 - attach_chunk_metadata
def attach_chunk_metadata(chunks, source):
    # TODO: wrap each chunk string with source, position, and chunk_id metadata.
    
    chunks_meta = []

    for position in range(len(chunks)):
        chunk_meta = {'text': chunks[position],
                        'source': source,
                        'position': position,
                        'chunk_id': f'{source}::{position}'
                        }
        chunks_meta.append(chunk_meta)

    return chunks_meta

# Step 11 - load_embedding_model
from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name):
    # TODO: return a sentence-transformers model instance for the given model_name.
    model = SentenceTransformer(model_name)

    return model

# Step 12 - embed_text
def embed_text(model, text):
    # TODO: Return a 1D float32 numpy embedding vector for the given text string.
    return model.encode(text)

# Step 13 - embed_chunks
def embed_chunks(model, chunks, batch_size=32):
    """Batch-embed a list of chunk strings or chunk dicts into a 2D float32 matrix."""
    # TODO: normalize chunk inputs to strings, encode in batches, return (n, d) float32 array
    
    if not chunks:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    
    embeddings = []

    for chunk in chunks:
        if isinstance(chunk, dict):
            embeddings.append(model.encode(chunk['text']))
        else:
            embeddings.append(model.encode(chunk))

    return np.array(embeddings)

# Step 14 - l2_normalize
import numpy as np

def l2_normalize(matrix):
    # TODO: rescale each row of `matrix` to unit L2 norm, leaving all-zero rows unchanged.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    norms = np.where(norms ==0, 1, norms)

    return matrix / norms

# Step 15 - save_corpus
import json

def save_corpus(embeddings, chunks, directory):
    # TODO: persist embeddings (.npy) and chunks (.json) into directory, then reload and return both.
    
    os.makedirs(directory, exist_ok=True)

    np.save(f"{directory}/embeddings.npy", embeddings)
    
    with open(f"{directory}/chunks.json", "w") as file:
        json.dump(chunks, file)

    loaded_embeddings = np.load(f"{directory}/embeddings.npy")
    
    with open(f"{directory}/chunks.json", "r") as file:
        loaded_chunks = json.load(file)

    return {
        "embeddings": loaded_embeddings,
        "chunks": loaded_chunks
    }

# Step 16 - cosine_similarity_search
import numpy as np

def cosine_similarity_search(query_vector, chunk_matrix):
    """Cosine similarity between query_vector (d,) and each row of chunk_matrix (n,d)."""
    # TODO: compute cosine similarity between the query vector and every chunk row

    query_norm = np.linalg.norm(query_vector)
    matrix_norm = np.linalg.norm(chunk_matrix, axis=1)

    scores = (chunk_matrix @ query_vector) / (query_norm * matrix_norm)

    return scores

# Step 17 - top_k_indices
import numpy as np

def top_k_indices(scores, k):
    """Return indices of the k highest scores in descending order."""
    # TODO: rank the score array and return the top-k positions as a numpy array

    indices = np.argsort(-scores, kind="stable")

    return indices[:k]

# Step 18 - top_k_chunks
import numpy as np

def top_k_chunks(scores, chunks, k):
    # TODO: return list of (chunk, score) tuples for the top-k scores, sorted descending

    top_k = top_k_indices(scores, k)

    chunk_score = []
    for i in top_k:
        chunk_score.append((chunks[i], float(scores[i])))

    return chunk_score

# Step 19 - retrieve
def retrieve(query, model, chunk_matrix, chunks, k):
    # TODO: embed the query, score it against chunk_matrix, return top-k (chunk, score) pairs.
    
    embed_query = embed_text(model, query)

    scores = cosine_similarity_search(embed_query, chunk_matrix)

    chunk_score = top_k_chunks(scores, chunks, k)

    return chunk_score

# Step 20 - build_faiss_index
def build_faiss_index(chunk_matrix):
    # TODO: build a FAISS inner-product index and add all rows of chunk_matrix to it
    
    d = chunk_matrix.shape[1]

    index = faiss.IndexFlatIP(d)

    index.add(chunk_matrix.astype(np.float32))

    return index

# Step 21 - faiss_search
import numpy as np
import faiss

def faiss_search(index, query_vector, k):
    """Return top-k (scores, indices) as 1D arrays for a single query vector."""
    # TODO: query the FAISS index with the single query vector and return flat top-k arrays
    
    query_vector = query_vector.reshape(1, -1).astype(np.float32)

    scores, indices = index.search(query_vector, k)

    scores = scores[0].astype(np.float32)
    indices = indices[0].astype(np.int64)

    return scores, indices

# Step 22 - compare_faiss_to_numpy
def compare_faiss_to_numpy(query_vector, chunk_matrix, index, k):
    # TODO: return True iff FAISS and numpy cosine search agree on the top-k indices

    scores = cosine_similarity_search(query_vector, chunk_matrix)
    numpy_indices = top_k_indices(scores, k)

    # FAISS retrieval
    _, faiss_indices = faiss_search(index, query_vector, k)

    # Compare sets of indices
    return set(numpy_indices) == set(faiss_indices)

# Step 23 - save_faiss_index
def save_faiss_index(index, path):
    """Write `index` to `path` and return the index loaded back from disk."""
    # TODO: persist the index to `path` and reload it; return the reloaded index
    faiss.write_index(index, path)

    ix = faiss.read_index(path)

    return ix

# Step 24 - build_prompt_template
def build_prompt_template():
    # TODO: return a RAG prompt template string with {context} and {question} placeholders.
    return (
        "Answer the question using only the provided context.\n\n"
        "Context: {context}\n\n"
        "Question: {question}"
    )

# Step 25 - format_context
def format_context(retrieved):
    # TODO: render each (chunk, score) as '[i] {text} (source={source})' and join with newlines
    text_source = []
    for i in range(len(retrieved)):
        data, score = retrieved[i]
        text = data["text"]
        source = data["source"]
        text_source.append(f"[{i+1}] {text} (source={source})")

    return "\n".join(text_source)

# Step 26 - truncate_context
def truncate_context(context, max_chars):
    # TODO: trim context so len(result) <= max_chars, preferring a whitespace boundary

    if len(context) <= max_chars:
        return context    
    
    truncate = context[:max_chars]
    ix = truncate.rfind(" ")

    if context[max_chars] == " ":
        return truncate

    if ix > 0:
        return truncate[:ix]
    else:
        return truncate[:max_chars]

# Step 27 - add_system_instruction
def add_system_instruction(prompt):
    """Prepend a fixed system instruction to the prompt."""
    # TODO: return a string that starts with a system instruction telling the model to use only the context
    
    system_prompt = "You are a helpful assistant. Answer the question using ONLY the provided context. If the answer is not in the context, say 'I do not know'."

    return system_prompt + "\n\n" + prompt

# Step 28 - load_generator
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_generator(model_name='sshleifer/tiny-gpt2'):
    # TODO: load a small local causal LM and its tokenizer, ensuring tokenizer.pad_token is set.
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# Step 29 - generate_answer
import torch

def generate_answer(model, tokenizer, prompt, max_new_tokens=32):
    # TODO: greedy-decode a continuation from `prompt` and return only the new tokens as a string.
    
    torch.manual_seed(42)

    inputs = tokenizer(prompt, return_tensors="pt")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    prompt_len = len(inputs["input_ids"][0])

    generated_ids = output_ids[0][prompt_len:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

# Step 30 - rag_answer
def rag_answer(query, chunks, embeddings, embed_model, generator, tokenizer, k=3):
    # TODO: embed query, retrieve top-k chunks, build prompt, generate answer, return dict.
    
    query_embedding = embed_model.encode(query)

    scores = cosine_similarity_search(
        query_embedding,
        embeddings
    )

    retrieved_chunks = top_k_chunks(
        scores,
        chunks,
        k
    )

    sources = []
    for chunk, score in retrieved_chunks:
        sources.append(chunk)

    chunks_formatted = format_context(retrieved_chunks)

    template = build_prompt_template()

    template.format(
        context=chunks_formatted,
        question=query
    )

    prompt = add_system_instruction(template)

    print(prompt)
    
    answer = generate_answer(
        generator,
        tokenizer,
        prompt
    )

    return {
        "answer": answer,
        "sources": sources,
        "query": query
    }

# Step 31 - track_source_chunk_ids
def track_source_chunk_ids(source_chunks):
    # TODO: return the list of chunk ids from the retrieved source chunks, preserving order
    ids = []

    for source in source_chunks:
        if 'id' in source.keys():
            ids.append(source['id'])

    return ids

# Step 32 - append_source_references
def append_source_references(answer_text, source_chunks):
    # TODO: append a 'Sources: [id1, id2, ...]' line to answer_text using the source chunk ids
    
    ids = track_source_chunk_ids(source_chunks)

    answer = f"{answer_text}\nSources: {str(ids).replace("'", "")}"

    return answer

# Step 33 - query_rewrite
def query_rewrite(raw_query):
    # TODO: clean and normalize a raw user query into a better search query
    
    query = normalize_text(raw_query).lower()

    phrases = ["please", "could you", "can you", "tell me", "i want to know"]

    changed = True
    while changed:
        changed = False
        for phrase in phrases:
            if query.startswith(phrase + " "):
                query = query[len(phrase):].strip()
                changed = True

    query = query.rstrip("?.!")
    
    return query

# Step 34 - hyde_retrieve
def hyde_retrieve(query, hypothetical_answer, chunks, embeddings, embed_model, k=5):
    # TODO: embed the hypothetical answer and return the top-k chunks by cosine similarity.
    
    hypothetical_answer_vector = embed_model.encode(hypothetical_answer)
    scores = cosine_similarity_search(hypothetical_answer_vector, embeddings)

    retrieved_chunks = top_k_chunks(scores, chunks, k)

    top_chunks = []

    for chunk, score in retrieved_chunks:
        top_chunks.append(chunk)
    
    return top_chunks

# Step 35 - reciprocal_rank_fusion
def reciprocal_rank_fusion(ranked_lists, k=60):
    # TODO: merge ranked lists of ids into one (id, score) list sorted by fused score.
    
    d = {}
    for lst in ranked_lists:
        for i in range(len(lst)):
            if lst[i] in d:
                d[lst[i]] += 1 / (k + i + 1)
            else:
                d[lst[i]] = 1 / (k + i + 1)

    sorted_dict = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
    
    out = []
    for k in sorted_dict:
        out.append((k, sorted_dict[k]))

    return out

# Step 36 - bm25_search
import math

def bm25_search(query, chunks, k=5, k1=1.5, b=0.75):
    # TODO: score chunks against the query with BM25 and return top-k (index, score) pairs

    query_terms = query.lower().split()

    docs = [chunk['text'].lower().split() for chunk in chunks]

    N = len(docs)

    if N == 0:
        return []

    avgdl = sum(len(doc) for doc in docs) / N

    # Document frequencies
    df = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    results = []

    for doc_idx, doc in enumerate(docs):
        doc_len = len(doc)

        # Term frequencies
        tf = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1

        score = 0.0

        for term in query_terms:
            if term not in tf:
                continue

            term_df = df.get(term, 0)

            idf = math.log(
                ((N - term_df + 0.5) / (term_df + 0.5)) + 1
            )

            freq = tf[term]

            score += idf * (
                freq * (k1 + 1)
            ) / (
                freq + k1 * (1 - b + b * doc_len / avgdl)
            )

        if score > 0:
            results.append((doc_idx, score))

    results.sort(key=lambda x: x[1], reverse=True)

    return results[:k]

# Step 37 - hybrid_search
def hybrid_search(query, chunks, embeddings, embed_model, alpha=0.5, k=5):
    # TODO: blend normalized dense cosine scores with BM25 scores and return the top-k (idx, score) pairs.

    query_embedding = embed_text(embed_model, query)

    dense_scores = cosine_similarity_search(
        query_embedding,
        embeddings
    )

    bm25_vec = np.zeros(len(chunks), dtype=float)

    for idx, score in bm25_search(
        query,
        chunks,
        k=len(chunks)
    ):
        bm25_vec[idx] = score

    def minmax(x):
        mn = np.min(x)
        mx = np.max(x)

        if mx == mn:
            return np.zeros_like(x, dtype=float)

        return (x - mn) / (mx - mn)

    dense_norm = minmax(dense_scores)
    bm25_norm = minmax(bm25_vec)

    combined = (
        alpha * dense_norm
        + (1 - alpha) * bm25_norm
    )

    ranked = sorted(
        enumerate(combined),
        key=lambda x: (-x[1], x[0])
    )

    
    return [
        (idx, float(score))
        for idx, score in ranked[:min(k, len(chunks))]
    ]

# Step 38 - rerank_cross_encoder
def rerank_cross_encoder(query, candidate_chunks, cross_encoder):
    # TODO: score (query, chunk) pairs with cross_encoder and return chunks sorted by descending score

    pairs = [
        (query, chunk["text"])
        for chunk in candidate_chunks
    ]

    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(candidate_chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [chunk for chunk, _ in ranked]

# Step 39 - maximal_marginal_relevance
def maximal_marginal_relevance(query_embedding, candidate_embeddings, k=5, lambda_param=0.5):
    # TODO: greedily pick indices balancing query relevance and diversity from already-selected items.

    n = len(candidate_embeddings)

    if n == 0:
        return []

    k = min(k, n)

    # Query-to-candidate similarities
    relevance = candidate_embeddings @ query_embedding

    selected = []
    remaining = set(range(n))

    while len(selected) < k:
        best_idx = None
        best_score = None

        for idx in remaining:

            if not selected:
                diversity_penalty = 0.0
            else:
                sims = candidate_embeddings[selected] @ candidate_embeddings[idx]
                diversity_penalty = np.max(sims)

            score = (
                lambda_param * relevance[idx]
                - (1 - lambda_param) * diversity_penalty
            )

            # Tie-break by smaller index
            if (
                best_score is None
                or score > best_score
                or (
                    score == best_score
                    and idx < best_idx
                )
            ):
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected

# Step 40 - filter_by_metadata
def filter_by_metadata(chunks, filter_dict):
    # TODO: return only chunks whose metadata contains every key/value pair in filter_dict

    if not filter_dict:
        return chunks

    result = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        keep = True
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                keep = False
                break

        if keep:
            result.append(chunk)

    return result

# Step 41 - build_eval_set
def build_eval_set():
    # TODO: return a small list of dicts with keys question, answer, relevant_ids

    return [
        {
            "question": "What is RAG?",
            "answer": "Retrieval-Augmented Generation combines a retriever with a generator.",
            "relevant_ids": ['c1', 'c2']
        },
        {
            "question": "What does FAISS do?",
            "answer": "FAISS performs fast nearest-neighbor search over dense vectors.",
            "relevant_ids": ['c3']
        },
        {
            "question": "Why normalize embeddings?",
            "answer": "So that inner products equal cosine similarities.",
            "relevant_ids": ['c4', 'c5']
        },
        {
            "question": "What is BM25?",
            "answer": "A lexical ranking function based on term frequency and document length.",
            "relevant_ids": ['c6']
        }
    ]

# Step 42 - hit_rate_at_k
def hit_rate_at_k(retrieved_ids_per_query, relevant_ids_per_query, k):
    # TODO: return the fraction of queries with at least one relevant id in the top-k retrieved

    if len(retrieved_ids_per_query) == 0:
        return 0.0

    hits = 0

    for i in range(len(retrieved_ids_per_query)):

        retrieved = retrieved_ids_per_query[i][:k]
        relevant = relevant_ids_per_query[i]

        found = False

        for doc_id in retrieved:
            if doc_id in relevant:
                found = True
                break

        if found:
            hits += 1

    return hits / len(retrieved_ids_per_query)

# Step 43 - recall_at_k
def recall_at_k(retrieved_ids_per_query, relevant_ids_per_query, k):
    # TODO: average over queries the fraction of relevant ids found in the top-k retrieved ids

    if not retrieved_ids_per_query:
        return 0.0

    total_recall = 0.0

    for retrieved, relevant in zip(
        retrieved_ids_per_query,
        relevant_ids_per_query
    ):
        if not relevant:
            recall = 0.0
        else:
            retrieved_top_k = set(retrieved[:k])
            relevant_set = set(relevant)

            recall = (
                len(retrieved_top_k & relevant_set)
                / len(relevant_set)
            )

        total_recall += recall

    return total_recall / len(retrieved_ids_per_query)

# Step 44 - mean_reciprocal_rank
def mean_reciprocal_rank(retrieved_ids_per_query, relevant_ids_per_query):
    # TODO: average the reciprocal rank of the first relevant id across queries

    if not retrieved_ids_per_query:
        return 0.0

    total_rr = 0.0

    for retrieved, relevant in zip(
        retrieved_ids_per_query,
        relevant_ids_per_query
    ):
        rr = 0.0

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break

        total_rr += rr

    return total_rr / len(retrieved_ids_per_query)

# Step 45 - faithfulness_score
def faithfulness_score(answer, context_chunks):
    # TODO: return the fraction of answer tokens that appear in the context text

    answer_tokens = normalize_text(answer).lower().split()

    if not answer_tokens:
        return 0.0

    
    context_text = " ".join(
        chunk["text"] for chunk in context_chunks
    )

    context_tokens = set(
        normalize_text(context_text).lower().split()
    )

    supported = 0

    for token in answer_tokens:
        if token in context_tokens:
            supported += 1

    return supported / len(answer_tokens)

# Step 46 - relevance_score
def relevance_score(answer, question):
    # TODO: return token-overlap (Jaccard) similarity between answer and question in [0, 1]

    punctuations = [",", ".", "!", "?"]

    for p in punctuations:
        if p in answer:
            answer = answer.replace(p, "")
        if p in question:
            question = question.replace(p, "")


    answer_tokens = set(
        normalize_text(answer).lower().split()
    )

    question_tokens = set(
        normalize_text(question).lower().split()
    )

    union = answer_tokens | question_tokens

    if not union:
        return 0.0

    intersection = answer_tokens & question_tokens

    return len(intersection) / len(union)

# Step 47 - handle_no_context
def handle_no_context(scored_chunks, threshold=0.2):
    """Return {'abstain': bool, 'message': str} based on top score vs threshold."""
    # TODO: abstain when no chunk's score strictly exceeds the threshold

    if not scored_chunks:
        return {
            "abstain": True,
            "message": "I do not know"
        }

    flag = True
    message = "I do not know"
    
    if isinstance(scored_chunks[0], dict):
        for chunk_dict in scored_chunks:
            if chunk_dict["score"] > threshold:
                flag = False
                message = ""
    else:    
        for _, score in scored_chunks:
            if score > threshold:
                flag = False
                message = ""

    return {
        "abstain": flag,
        "message": message
    }

# Step 48 - deduplicate_chunks (not yet solved)
# TODO: implement

# Step 49 - cache_query_embedding (not yet solved)
# TODO: implement

# Step 50 - update_chat_memory (not yet solved)
# TODO: implement

# Step 51 - rewrite_followup (not yet solved)
# TODO: implement

