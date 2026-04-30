from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import openai
import torch
import torch.nn.functional as F
from ot.backend import get_backend

MOTIVATION_DIR = Path(__file__).resolve().parent
QUESTION_PROMPT_FILE = MOTIVATION_DIR / "question.txt"
PROBLEM_PROMPT_FILE = MOTIVATION_DIR / "problem.txt"


def _read_prompt(prompt_file: Path) -> str:
    return prompt_file.read_text(encoding="utf-8")


def get_vecs(model, tokenizer, sentence, device):
    model.eval()
    max_length = getattr(tokenizer, "model_max_length", 512)
    if not isinstance(max_length, int) or max_length <= 0 or max_length > 512:
        max_length = 512
    encoded_input = tokenizer(sentence, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded_input = {_key: encoded_input[_key].to(device) for _key in encoded_input}
    with torch.no_grad():
        model_output = model(**encoded_input)
        cls_vecs = model_output[0][:, 0]
    cls_vecs = torch.nn.functional.normalize(cls_vecs, p=2, dim=1)
    word_vecs = model_output[0]
    return cls_vecs, word_vecs


def compute_weights_uniform(s1_word_embeddigs, s2_word_embeddigs):
    s1_weights = torch.ones(s1_word_embeddigs.shape[0], dtype=torch.float64, device=s1_word_embeddigs.device)
    s2_weights = torch.ones(s2_word_embeddigs.shape[0], dtype=torch.float64, device=s2_word_embeddigs.device)
    return s1_weights, s2_weights


def compute_distance_matrix_cosine(s1_word_embeddigs, s2_word_embeddigs, distortion_ratio):
    C = (torch.matmul(F.normalize(s1_word_embeddigs), F.normalize(s2_word_embeddigs).t()) + 1.0) / 2
    C = apply_distortion(C, distortion_ratio)
    C = min_max_scaling(C)
    C = 1.0 - C
    return C


def min_max_scaling(C):
    eps = 1e-10
    nx = get_backend(C)
    C_min = nx.min(C)
    C_max = nx.max(C)
    C = (C - C_min + eps) / (C_max - C_min + eps)
    return C


def apply_distortion(sim_matrix, ratio):
    shape = sim_matrix.shape
    if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
        return sim_matrix

    pos_x = torch.tensor(
        [[y / float(shape[1] - 1) for y in range(shape[1])] for x in range(shape[0])],
        device=sim_matrix.device,
        dtype=sim_matrix.dtype,
    )
    pos_y = torch.tensor(
        [[x / float(shape[0] - 1) for x in range(shape[0])] for y in range(shape[1])],
        device=sim_matrix.device,
        dtype=sim_matrix.dtype,
    )
    distortion_mask = 1.0 - ((pos_x - pos_y.T) ** 2) * ratio
    sim_matrix = torch.mul(sim_matrix, distortion_mask)
    return sim_matrix


def _openai_client(api_key: str, base_url: str):
    return openai.OpenAI(api_key=api_key, base_url=base_url)


class FolGenerationCache:
    def __init__(self, cache_path: str | Path | None = None):
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}

        if self.cache_path and self.cache_path.exists():
            print(f"Loading NS-IR cache from {self.cache_path}...")
            with self.cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    kind = row.get("kind")
                    text = row.get("text")
                    if not kind or not text:
                        continue
                    self._cache[self._make_key(kind, text)] = row

    @staticmethod
    def _make_key(kind: str, text: str) -> str:
        digest = hashlib.sha1(f"{kind}\0{text}".encode("utf-8")).hexdigest()
        return f"{kind}:{digest}"

    def get(self, kind: str, text: str) -> dict | None:
        return self._cache.get(self._make_key(kind, text))

    def set(self, kind: str, text: str, payload: dict) -> None:
        record = dict(payload)
        record.setdefault("kind", kind)
        record.setdefault("text", text)
        record.setdefault("cache_key", self._make_key(kind, text))

        key = self._make_key(kind, text)
        self._cache[key] = record

        if self.cache_path is None:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_fol_body(generated_text: str, section_label: str) -> str:
    if "Predicates:" not in generated_text or section_label not in generated_text:
        return ""

    try:
        after_predicates = generated_text.split("Predicates:", 1)[1]
        body = after_predicates.split(section_label, 1)[0]
        return body.strip()
    except IndexError:
        return ""


def _generate_fol_text(client, prompt_input: str, section_label: str, kind: str, model_name: str, wait_till_success: bool = False) -> str:
    retry_prompt = (
        f"Invalid format. Re-output the {kind} using only Predicates: and {section_label}, "
        f"one formula per line, and nothing else."
    )

    for attempt in range(2):
        current_prompt = prompt_input if attempt == 0 else f"{prompt_input}\n\n{retry_prompt}"
        request_params = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem.",
                },
                {"role": "user", "content": current_prompt},
            ],
        }
        request_params["model"] = model_name
        try:
            response = client.chat.completions.create(**request_params)
            response = response.model_dump()
            generated_text = response["choices"][0]["message"]["content"].strip()
            body = _extract_fol_body(generated_text, section_label)
            if body:
                return generated_text
        except Exception:
            if attempt == 0 or wait_till_success:
                time.sleep(1.0)
                continue
            raise

    raise ValueError(f"Unable to parse {kind} FOL output after retry: {generated_text}")


def query2fol(
    query: str,
    args,
    cache: FolGenerationCache | None = None,
):
    cached = cache.get("query", query) if cache else None
    if cached and cached.get("premise"):
        return cached["premise"]

    client = _openai_client(args.api_key, args.base_url)
    prompt = _read_prompt(QUESTION_PROMPT_FILE)
    prompt_input = prompt.replace("%QUERY%", query)
    generated_text = _generate_fol_text(
        client,
        prompt_input,
        "Conclusion:",
        "query",
        getattr(args, "generator_model", "gpt-4o"),
        getattr(args, "wait_till_success", False),
    )
    query_premise_body = _extract_fol_body(generated_text, "Conclusion:")
    if not query_premise_body:
        raise ValueError(f"Unable to parse query FOL output after retry: {generated_text}")
    query_premise = " ".join([q.split(" ::: ")[0] for q in query_premise_body.split("\n")])
    if cache:
        cache.set(
            "query",
            query,
            {
                "query": query,
                "premise": query_premise,
                "generator_model": getattr(args, "generator_model", "gpt-4o"),
                "base_url": args.base_url,
            },
        )
    # print("************generate FOL-query************")
    return query_premise


def doc2fol(
    document: str,
    args,
    cache: FolGenerationCache | None = None,
):
    cached = cache.get("doc", document) if cache else None
    if cached and cached.get("premise"):
        return cached["premise"]

    client = _openai_client(args.api_key, args.base_url)
    prompt = _read_prompt(PROBLEM_PROMPT_FILE)
    prompt_input = prompt.replace("%DOCUMENT%", document)
    generated_text = _generate_fol_text(
        client,
        prompt_input,
        "Premises:",
        "document",
        getattr(args, "generator_model", "gpt-4o"),
        getattr(args, "wait_till_success", False),
    )
    doc_premise_body = _extract_fol_body(generated_text, "Premises:")
    if not doc_premise_body:
        raise ValueError(f"Unable to parse document FOL output after retry: {generated_text}")
    doc_premise = " ".join([q.split(" ::: ")[0] for q in doc_premise_body.split("\n")])
    if cache:
        cache.set(
            "doc",
            document,
            {
                "text": document,
                "premise": doc_premise,
                "generator_model": getattr(args, "generator_model", "gpt-4o"),
                "base_url": args.base_url,
            },
        )
    # print("************generate FOL-document************")
    return doc_premise


def get_cos_score(query, document):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    return (cos(query, document) + 1) / 2


def updated_embeddings(
    model,
    tokenizer,
    text,
    premise,
    device,
    distortion_ratio=0.2,
    sinkhorn=False,
    epsilon=0.1,
    stop_thr=1e-6,
    num_itermax=1000,
):
    text_cls_vecs, text_word_vecs = get_vecs(model, tokenizer, text, device)
    premise_cls_vecs, premise_word_vecs = get_vecs(model, tokenizer, premise, device)
    text_cls_vecs = text_cls_vecs.squeeze(0)
    text_word_vecs, premise_word_vecs = text_word_vecs.squeeze(0), premise_word_vecs.squeeze(0)

    C = compute_distance_matrix_cosine(text_word_vecs, premise_word_vecs, distortion_ratio)
    # print("**************Logic Alignment****************")
    text_word_weights, premise_word_weights = compute_weights_uniform(text_word_vecs, premise_word_vecs)
    text_word_weights = text_word_weights / text_word_weights.sum()
    premise_word_weights = premise_word_weights / premise_word_weights.sum()
    text_word_weights = text_word_weights.cpu().numpy()
    premise_word_weights = premise_word_weights.cpu().numpy()
    C = C.cpu().numpy()

    import ot

    if sinkhorn:
        P = ot.bregman.sinkhorn_log(
            text_word_weights,
            premise_word_weights,
            C,
            reg=epsilon,
            stopThr=stop_thr,
            numItermax=num_itermax,
        )
    else:
        P = ot.emd(text_word_weights, premise_word_weights, C)
    P = torch.from_numpy(P).float().to(device)
    query_embedding = torch.einsum("md, mn, nd, d-> d", [text_word_vecs, P, premise_word_vecs, text_cls_vecs])
    m = text_word_vecs.shape[0]
    n = premise_word_vecs.shape[0]
    d = premise_word_vecs.shape[1]
    # print("**************Connective Constraint****************")
    tokens = tokenizer.tokenize(premise)
    tokens_ = []
    for token in tokens:
        if token == "¬":
            tokens_.append(-1)
        elif token == "∧" or token == "∨" or token == "⊕" or token == "→" or token == "↔" or token == "∀" or token == "∃":
            tokens_.append(1)
        else:
            tokens_.append(0)
    tokens_ = [0] + tokens_ + [0]
    tokens_ = tokens_[:512]
    tokens = torch.tensor(tokens_).to(device).unsqueeze(1).expand(-1, m).T.clone()
    tokens[((tokens == -1) & (P > 0)) | ((tokens == 1) & (P > 0))] = 0
    tokens = tokens.unsqueeze(2).expand(-1, -1, d)
    text_word_vecs_ = text_word_vecs.unsqueeze(1).expand(-1, n, -1)
    premise_word_vecs_ = premise_word_vecs.unsqueeze(0).expand(m, -1, -1)
    attention = torch.einsum("nd, mnd->nm", [premise_word_vecs, text_word_vecs_ + tokens * premise_word_vecs_])
    attention = torch.softmax(attention, dim=1)
    premise_text_ = torch.einsum("nm, mnd->nd", [attention, text_word_vecs_ + tokens * premise_word_vecs_])

    return query_embedding, premise_text_.mean(dim=0)
