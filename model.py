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

    answer = f"{answer_text}\nSources: {ids}"
    answer = answer.replace("'", "")

    return answer

# Step 33 - query_rewrite (not yet solved)
# TODO: implement

# Step 34 - hyde_retrieve (not yet solved)
# TODO: implement

# Step 35 - reciprocal_rank_fusion (not yet solved)
# TODO: implement

# Step 36 - bm25_search (not yet solved)
# TODO: implement

# Step 37 - hybrid_search (not yet solved)
# TODO: implement

# Step 38 - rerank_cross_encoder (not yet solved)
# TODO: implement

# Step 39 - maximal_marginal_relevance (not yet solved)
# TODO: implement

# Step 40 - filter_by_metadata (not yet solved)
# TODO: implement

# Step 41 - build_eval_set (not yet solved)
# TODO: implement

# Step 42 - hit_rate_at_k (not yet solved)
# TODO: implement

# Step 43 - recall_at_k (not yet solved)
# TODO: implement

# Step 44 - mean_reciprocal_rank (not yet solved)
# TODO: implement

# Step 45 - faithfulness_score (not yet solved)
# TODO: implement

# Step 46 - relevance_score (not yet solved)
# TODO: implement

# Step 47 - handle_no_context (not yet solved)
# TODO: implement

# Step 48 - deduplicate_chunks (not yet solved)
# TODO: implement

# Step 49 - cache_query_embedding (not yet solved)
# TODO: implement

# Step 50 - update_chat_memory (not yet solved)
# TODO: implement

# Step 51 - rewrite_followup (not yet solved)
# TODO: implement

