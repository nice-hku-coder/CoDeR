from __future__ import annotations

import re
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
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
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


def query2fol(
    query: str,
    args,
):
    client = _openai_client(args.api_key, args.base_url)
    prompt = _read_prompt(QUESTION_PROMPT_FILE)
    prompt_input = prompt.replace("%QUERY%", query)
    request_params = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem.",
            },
            {"role": "user", "content": prompt_input},
        ],
    }
    response = client.chat.completions.create(**request_params)
    response = response.model_dump()
    generated_text = response["choices"][0]["message"]["content"].strip()
    pattern = re.compile(r"(.*)(?=Predicates:)|(?<=Predicates:)(.*?)(?=Conclusion:)|(?<=Conclusion:)(.*?)(?=\Z)", re.DOTALL)
    matches = pattern.findall(generated_text)
    query_premise = " ".join([q.split(" ::: ")[0] for q in matches[2][2].strip().split("\n")])
    # print("************generate FOL-query************")
    return query_premise


def doc2fol(
    document: str,
    args,
):
    client = _openai_client(args.api_key, args.base_url)
    prompt = _read_prompt(PROBLEM_PROMPT_FILE)
    prompt_input = prompt.replace("%DOCUMENT%", document)
    request_params = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem.",
            },
            {"role": "user", "content": prompt_input},
        ],
    }
    response = client.chat.completions.create(**request_params)
    response = response.model_dump()
    generated_text = response["choices"][0]["message"]["content"].strip()
    pattern = re.compile(r"(.*)(?=Predicates:)|(?<=Predicates:)(.*?)(?=Premises:)|(?<=Premises:)(.*?)(?=\Z)", re.DOTALL)
    matches = pattern.findall(generated_text)
    doc_premise = " ".join([q.split(" ::: ")[0] for q in matches[2][2].strip().split("\n")])
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
