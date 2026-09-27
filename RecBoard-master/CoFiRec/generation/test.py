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
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie
from modeling_cofirec import CoFiRec


def test(args):

    set_seed(args.seed)
    print(vars(args))

    device_map = {"": args.gpu_id}
    device = torch.device("cuda",args.gpu_id)

    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model, model_max_length=512
    )
    train_data, _ = load_datasets(args)
    tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config = T5Config.from_pretrained(args.base_model)
    config.vocab_size = len(tokenizer)

    model = CoFiRec.from_pretrained(
        args.ckpt_path,
        config=config,
        low_cpu_mem_usage=True,
        device_map={"": args.gpu_id},
        ignore_mismatched_sizes=True,  
    )
    # model.set_hyper(args.temperature) 
    model.to(device).eval()
    prompt_ids = [0]
    test_data = load_test_dataset(args)

    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()


    candidate_trie = Trie(
        [
            [0] + tokenizer.encode(candidate)
            for candidate in all_items
        ]
    )
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=True, num_workers=4, pin_memory=True)


    print("data num:", len(test_data))

    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results = []
    with torch.no_grad():
        for prompt_id in prompt_ids:

            
            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                total += len(targets)
                if step == 0:
                    print(inputs)
                    print(targets)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]   
                    
                inputs_embeds = model.shared(input_ids) 
        
                seq_len, device = input_ids.size(1), input_ids.device
                pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(input_ids.size(0), -1)
                group_ids = (pos_ids // model.group_size)
                group_pos_emb = model.group_embedding(group_ids)
                mod_ids = (pos_ids % model.group_size)
                mod_pos_emb = model.pos_embedding(mod_ids)
                
                am = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)  # [B, L, 1]
                eos_id = model.config.eos_token_id
                non_eos = (input_ids != eos_id).unsqueeze(-1).to(inputs_embeds.dtype)
                position_embedding_mask = am * non_eos  # 1 for valid positions
                pos_embd =  group_pos_emb * position_embedding_mask + mod_pos_emb * position_embedding_mask
                # pos_embd =  group_pos_emb * position_embedding_mask
                
                # combine
                inputs_embeds = inputs_embeds + pos_embd

                output = model.generate(
                    # input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=10,
                    # max_length=10,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )
                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                topk_res = get_topk_results(output,scores,targets,args.num_beams,
                                            all_items=all_items if args.filter_items else None)

                batch_metrics_res = get_metrics_results(topk_res, metrics)
                # print(batch_metrics_res)

                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                # if (step+1)%10 == 0:
                temp={}
                for m in metrics_results:
                    temp[m] = metrics_results[m] / total
                print(temp)
            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total
            all_prompt_results.append(metrics_results)
            print("======================================================")
            print("Prompt {} results: ".format(prompt_id), metrics_results)
            print("======================================================")
            print("")
    
    mean_results = {}
    min_results = {}
    max_results = {}

    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res)/len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("======================================================")
    print("Mean results: ", mean_results)
    print("Min results: ", min_results)
    print("Max results: ", max_results)
    print("======================================================")


    save_data={}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results

    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    test(args)
