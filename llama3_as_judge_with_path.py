#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import torch
import csv
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Load external prompt template
# ============================================================
def load_prompt_template(path):
    with open(path, "r", encoding="utf8") as f:
        return f.read()

# ============================================================
# Automatic JSON / JSONL loader
# ============================================================
def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf8") as f:
        raw = f.read().strip()

    if raw.startswith("{"):
        return [json.loads(line) for line in raw.split("\n") if line.strip()]
    elif raw.startswith("["):
        return json.loads(raw)
    else:
        raise ValueError(f"Unrecognized file format: {path}")

# ============================================================
# Token groups (3 tiers)
#   1) pure: exact "1".."5"
#   2) digit_plus: strip -> digit or digit+punct (e.g., " 5", "5.", "\n4)")
#   3) digit_word: digit_plus + number words ("one".."five")
# ============================================================
def collect_1to5_token_sets(tokenizer):
    digits = ["1", "2", "3", "4", "5"]
    number_words = ["one", "two", "three", "four", "five"]

    pure = {d: [] for d in digits}
    digit_plus = {d: [] for d in digits}
    digit_word = {d: [] for d in digits}

    # Extend if you see other suffixes in generations
    punct_chars = r"\.\,\:\;\!\?\)\]\}\>\-–—…、，。：；！？"
    digit_punct_re = re.compile(rf"^([1-5])([{punct_chars}]*)$")

    word2digit = {w: str(i + 1) for i, w in enumerate(number_words)}

    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid])
        s_strip = s.strip().lower()

        # 1) pure
        if s in digits:
            pure[s].append(tid)

        # 2) digit_plus
        m = digit_punct_re.match(s_strip)
        if m is not None:
            d = m.group(1)
            digit_plus[d].append(tid)

        # 3) digit_word
        if m is not None:
            d = m.group(1)
            digit_word[d].append(tid)
        elif s_strip in word2digit:
            digit_word[word2digit[s_strip]].append(tid)

    return pure, digit_plus, digit_word

def get_single_pure_token_ids(PURE, tokenizer=None):
    """
    Return dict: {"1": tid1, ..., "5": tid5}
    Enforce exactly ONE token id per digit in PURE.
    (You said: no summing/aggregation; store 1..5 separately.)
    """
    out = {}
    for d in ["1", "2", "3", "4", "5"]:
        ids = PURE[d]
        if len(ids) != 1:
            decoded = [tokenizer.decode([tid]) for tid in ids] if tokenizer is not None else None
            raise ValueError(
                f"PURE['{d}'] has {len(ids)} token ids, expected 1. "
                f"ids={ids}, decoded={decoded}"
            )
        out[d] = ids[0]
    return out

# ============================================================
# Logits → aggregated probs over 1..5 only (5-way normalized)
# ============================================================
def compute_probs_from_logits(logits_1d, token_sets):
    keys = ["1", "2", "3", "4", "5"]
    logps = []
    for k in keys:
        tids = token_sets[k]
        if len(tids) == 0:
            logps.append(torch.tensor(-float("inf"), device=logits_1d.device))
        else:
            logps.append(torch.logsumexp(logits_1d[tids], dim=0))

    logps = torch.stack(logps)
    probs = torch.softmax(logps, dim=-1)
    return {k: float(probs[i].item()) for i, k in enumerate(keys)}

# ============================================================
# Pick prompt file based on dim
# ============================================================
def resolve_prompt_file(prompt_file_arg, dim):
    """
    Behavior:
    - If user passes --prompt_file explicitly, use it.
    - Else use default path: data/prompts/{dim}_detailed.txt
    """
    if prompt_file_arg is not None:
        return prompt_file_arg

    allowed_dims = ["coherence", "consistency", "fluency", "relevance"]
    if dim not in allowed_dims:
        raise ValueError(
            f"--dim must be one of {allowed_dims}, got '{dim}'"
        )

    return f"data/prompts/{dim}_detailed.txt"


# ============================================================
# Batch generation (ONLY generate; return scores)
# ============================================================
def batch_generate(model, tokenizer, list_of_messages, max_new_tokens):
    texts = tokenizer.apply_chat_template(
        list_of_messages,
        add_generation_prompt=True,
        tokenize=False
    )

    tokenized = tokenizer(
        texts,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.append(eot)

    with torch.no_grad():
        outputs = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_ids,
            return_dict_in_generate=True,
            output_scores=True
        )

    return outputs, tokenized

# ============================================================
# Extract rating logits + raw_text using ONLY generate scores
# (parameterized by token_sets)
# ============================================================
def batch_extract_rating_logits_from_generate(outputs, tokenized, token_sets, tokenizer):
    sequences = outputs.sequences
    total_steps = len(outputs.scores)
    bsz, seq_total_len = sequences.shape

    prompt_len = tokenized["input_ids"].shape[1]
    assert seq_total_len == prompt_len + total_steps

    all_rating_ids = {tid for ids in token_sets.values() for tid in ids}

    rating_logits_list = []
    found_flags = []
    rating_positions = []
    raw_texts = []

    for i in range(bsz):
        seq = sequences[i]
        gen_tokens = seq[prompt_len:]

        raw_texts.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))

        gen_step = None
        for offset, tok_id in enumerate(gen_tokens):
            if tok_id.item() in all_rating_ids:
                gen_step = offset
                break

        if gen_step is None:
            rating_logits_list.append(None)
            found_flags.append(0)
            rating_positions.append(None)
            continue

        # logits for the step right before emitting the rating token
        rating_logits_list.append(outputs.scores[gen_step][i])
        found_flags.append(1)
        rating_positions.append(gen_step)

    return rating_logits_list, found_flags, rating_positions, raw_texts

# ============================================================
# PURE trajectory extraction:
# record all step-wise distributions over 1-5 (PURE) until the
# first emitted PURE rating token step.
#
# Additionally store:
# - probs: 5-way normalized over {1..5} (your previous "only 1-5 softmax")
# - vocab_probs_single: full-vocab softmax probabilities for the SINGLE
#   token ids of "1","2","3","4","5" (no summing / no aggregation)
# ============================================================
def extract_pure_traj_from_generate(outputs, tokenized, PURE, PURE_SINGLE, tokenizer):
    sequences = outputs.sequences
    total_steps = len(outputs.scores)
    bsz, seq_total_len = sequences.shape

    prompt_len = tokenized["input_ids"].shape[1]
    assert seq_total_len == prompt_len + total_steps

    pure_ids_union = {tid for ids in PURE.values() for tid in ids}

    traj_list = []
    raw_texts = []

    for i in range(bsz):
        seq = sequences[i]
        gen_tokens = seq[prompt_len:]
        raw_texts.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))

        first_step = None
        for t, tok_id in enumerate(gen_tokens):
            if tok_id.item() in pure_ids_union:
                first_step = t
                break

        if first_step is None:
            traj_list.append({"found": 0, "rating_gen_step": None, "traj": []})
            continue

        traj = []
        # include the step where the rating token is emitted (t = first_step)
        for t in range(first_step + 1):
            logits_t = outputs.scores[t][i]  # (vocab,)

            # (A) 5-way normalized probs over 1..5 (aggregated within each digit)
            probs_1to5 = compute_probs_from_logits(logits_t, PURE)

            # (B) full-vocab softmax probs for the single token ids of "1".."5"
            probs_vocab = torch.softmax(logits_t, dim=-1)
            vocab_probs_single = {
                d: float(probs_vocab[PURE_SINGLE[d]].item())
                for d in ["1", "2", "3", "4", "5"]
            }

            traj.append({
                "step": t,
                "probs": probs_1to5,
                "vocab_probs_single": vocab_probs_single
            })

        traj_list.append({"found": 1, "rating_gen_step": first_step, "traj": traj})

    return traj_list, raw_texts

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default="data/summeval.json")
    # If omitted, we pick based on --dim (see resolve_prompt_file)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--dim",
        type=str,
        default="coherence",
        choices=["coherence", "consistency", "fluency", "relevance"]
    )

    parser.add_argument("--out", type=str, default="summeval_results_3tiers.csv")

    # NEW: pure trajectory output (JSONL)
    parser.add_argument("--pure_traj_out", type=str, default="pure_rating_traj.jsonl")

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)


    prompt_path = resolve_prompt_file(args.prompt_file, args.dim)

    # ============================================================
    # Auto-append dim to output filename
    # ============================================================
    out_name = args.out
    if out_name.endswith(".csv"):
        out_name = out_name[:-4]
    out_name = f"{out_name}_{args.dim}.csv"

    args.out = os.path.join("results", out_name)


    # Also append dim to pure traj filename
    traj_name = args.pure_traj_out
    if traj_name.endswith(".jsonl"):
        traj_name = traj_name[:-6]
    traj_name = f"{traj_name}_{args.dim}.jsonl"

    args.pure_traj_out = os.path.join("results", traj_name)


    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16
    ).cuda()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id

    PURE, DIGITPLUS, DIGITWORD = collect_1to5_token_sets(tokenizer)
    PURE_SINGLE = get_single_pure_token_ids(PURE, tokenizer)

    TOKENSETS = {
        "pure": PURE,
        "digit_plus": DIGITPLUS,
        "digit_word": DIGITWORD
    }

    PROMPT_TEMPLATE = load_prompt_template(prompt_path)

    data_list = load_json_or_jsonl(args.jsonl)
    if args.limit:
        data_list = data_list[:args.limit]

    with open(args.out, "w", newline="", encoding="utf8") as fcsv, \
         open(args.pure_traj_out, "w", encoding="utf8") as fpure:
        writer = csv.writer(fcsv)

        # Keep it similar to your original, but expanded for 3 tiers
        header = ["doc_id", "gold", "raw_text"]
        for name in TOKENSETS.keys():
            header += [
                f"{name}_found_rating_token",
                f"{name}_rating_gen_step",
                f"{name}_rating_probs_json",
                f"{name}_first_probs_json",
            ]
        writer.writerow(header)

        for i in tqdm(range(0, len(data_list), args.batch_size), desc="Evaluating"):
            batch = data_list[i:i + args.batch_size]
            messages, golds, ids = [], [], []

            for item in batch:
                prompt = PROMPT_TEMPLATE.format(
                    Document=item["source"],
                    Summary=item["system_output"]
                )
                messages.append([
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ])
                golds.append(item["scores"][args.dim])
                ids.append(item["doc_id"])

            outputs, tokenized = batch_generate(
                model, tokenizer, messages, args.max_new_tokens
            )

            # First-step logits come from generate scores (no extra forward)
            # outputs.scores[0]: (bsz, vocab)
            first_logits = outputs.scores[0] if len(outputs.scores) > 0 else None

            # PURE trajectories (step-wise 1-5 probs until first pure rating token)
            pure_traj_list, pure_raw_texts = extract_pure_traj_from_generate(
                outputs, tokenized, PURE, PURE_SINGLE, tokenizer
            )

            # Extract rating logits for each tier (from generate scores)
            rating_pack = {}
            raw_texts = None
            for name, ts in TOKENSETS.items():
                r_logits, found, r_pos, r_texts = batch_extract_rating_logits_from_generate(
                    outputs, tokenized, ts, tokenizer
                )
                rating_pack[name] = (r_logits, found, r_pos)
                if raw_texts is None:
                    raw_texts = r_texts  # same across tiers

            # Write rows + write JSONL per sample
            for j in range(len(batch)):
                row = [ids[j], golds[j], raw_texts[j]]

                for name, ts in TOKENSETS.items():
                    r_logits, found, r_pos = rating_pack[name]

                    if found[j]:
                        rating_probs = compute_probs_from_logits(r_logits[j], ts)
                    else:
                        rating_probs = {}

                    if first_logits is None:
                        first_probs = {}
                    else:
                        first_probs = compute_probs_from_logits(first_logits[j], ts)

                    row += [
                        found[j],
                        r_pos[j],
                        json.dumps(rating_probs, ensure_ascii=False),
                        json.dumps(first_probs, ensure_ascii=False),
                    ]

                writer.writerow(row)

                # JSONL record for PURE trajectory (per sample)
                pure_rec = {
                    "doc_id": ids[j],
                    "dim": args.dim,
                    "gold": golds[j],
                    "found_pure_rating_token": pure_traj_list[j]["found"],
                    "pure_rating_gen_step": pure_traj_list[j]["rating_gen_step"],
                    "pure_traj": pure_traj_list[j]["traj"],  # list[{step, probs, vocab_probs_single}]
                    "raw_text": raw_texts[j],
                }
                fpure.write(json.dumps(pure_rec, ensure_ascii=False) + "\n")

    print("Saved CSV:", args.out)
    print("Saved PURE traj JSONL:", args.pure_traj_out)

if __name__ == "__main__":
    main()
