import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
from typing import List

import torch
import transformers
# from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration

from utils import *
from collator import Collator, TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie

def extract_beam_outputs(args):
    set_seed(args.seed)
    device = torch.device("cuda", args.gpu_id)
    tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        low_cpu_mem_usage=True,
        device_map={"": args.gpu_id},
    )
    model.eval()

    train_data, _ = load_datasets(args)
    tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())

    collator = TestCollator(args, tokenizer)
    train_loader = DataLoader(
        train_data,
        batch_size=1,
        collate_fn=collator,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # train_dataset = map_train_dataset(args, train_data)
    all_items = train_data.datasets[0].get_all_items()
    candidate_trie = Trie([[0] + tokenizer.encode(item) for item in all_items])
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    beams = 50
    output_path = "train_beam_outputs.jsonl"
    count = 0

    with open(output_path, "w", encoding="utf-8") as f, torch.no_grad():
        for batch in tqdm(train_loader, desc="Train Beam Extraction"):
            inputs, labels = batch
            print('input:',inputs)
            print('labels:',labels)
            import pdb; pdb.set_trace()
            inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_out = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=50,
                num_return_sequences=beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )

            # 1) Get ids, scores, and decode to text
            output_ids    = gen_out.sequences            # Tensor [B*beams, L]
            output_scores = gen_out.sequences_scores     # Tensor [B*beams]
            output_texts  = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )  # List[str]

            B = inputs["input_ids"].size(0)
            L_gen = output_ids.size(1)

            # 2) Process each sample
            for i in range(B):
                start = i * beams
                end   = start + beams

                sample_ids    = output_ids[start:end]        # Tensor [beams, L_gen]
                sample_scores = output_scores[start:end].tolist()
                sample_texts  = output_texts[start:end]      # List[str]
                gold_ids      = labels[i]                    # List[int]

                # 2.1 Sort by score
                idx = sorted(range(beams), key=lambda j: sample_scores[j], reverse=True)
                sorted_texts  = [sample_texts[j] for j in idx]
                sorted_scores = [sample_scores[j] for j in idx]
                sorted_ids    = sample_ids[idx, :]           # Tensor [beams, L_gen]

                # 2.2 Collect negative tokens per position
                neg_tokens_per_pos = []
                for pos in range(min(len(gold_ids), L_gen)):
                    unique_ids = set(sorted_ids[:, pos].tolist())
                    unique_ids.discard(gold_ids[pos])
                    neg_tokens = tokenizer.convert_ids_to_tokens(list(unique_ids))
                    neg_tokens_per_pos.append(neg_tokens)

                # 3) Write record
                record = {
                    "labels":               gold_ids,
                    "beam_seqs":            sorted_texts,
                    "beam_scores":          sorted_scores,
                    "neg_tokens_per_pos":   neg_tokens_per_pos,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                count += 1

    print(f"Streamed {count} samples to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    from utils import parse_global_args, parse_dataset_args, parse_test_args
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    args = parser.parse_args()

    extract_beam_outputs(args)
