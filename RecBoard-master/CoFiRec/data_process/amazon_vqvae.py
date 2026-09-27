import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from utils import set_device, load_json, load_plm, clean_text
from transformers import AutoTokenizer, AutoModel
import re
import html

def clean_text(raw_text):
    if isinstance(raw_text, list):
        new_raw_text=[]
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r'</?\w+[^>]*>', '', raw)
            raw = re.sub(r'["\n\r]*', '', raw)
            new_raw_text.append(raw.strip())
        cleaned_text = ' '.join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r'</?\w+[^>]*>', '', cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text


def load_data(args):
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    return load_json(item2feature_path)


def generate_text(item2feature):
    item_data = []
    
    for item_id, data in item2feature.items():
        category_brand = clean_text(f"{data.get('category', '')} {data.get('brand', '')}").strip()
        title = clean_text(data.get('title', '')).strip()
        description = clean_text(data.get('description', '')).strip()
        
        if description in ["", "."]:
            description = title
            title_w_desc = title
        else:
            title_w_desc = f"{title} {description}"

        item_data.append((int(item_id), category_brand, title, description, title_w_desc))

    return item_data

def generate_text_abaltion(item2feature):
    item_data = []
    hierarchy_path = os.path.join(args.root, f"{args.dataset}.item.json")
    hierarchy_data = load_json(hierarchy_path)

    for item_id, data in hierarchy_data.items():
        category_brand = f"{data[0]} {data[1]}".strip()
        title = data[2].strip()
        description = data[3].strip()
        item_data.append((int(item_id), category_brand, title, description))

    return item_data

def generate_embedding(args, item_data, tokenizer, model):
    print('Generating embeddings for category+brand, title, and description:')
    
    embeddings = []
    batch_size = 1

    def encode_texts(text_list):
        tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(text_list, max_length=args.max_sent_len, truncation=True,
                            return_tensors='pt', padding="longest").to(args.device)
        with torch.no_grad():
            outputs = model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask)
        masked_output = outputs.last_hidden_state * encoded.attention_mask.unsqueeze(-1)
        mean_output = masked_output.sum(dim=1) / encoded.attention_mask.sum(dim=-1, keepdim=True)
        return mean_output.cpu()

    for start in tqdm(range(0, len(item_data), batch_size)):
        batch = item_data[start: start + batch_size]

        category_brand_texts = [x[1] for x in batch]
        title_texts = [x[2] for x in batch]
        description_texts = [x[3] for x in batch]
        title_w_desc_texts = [x[4] for x in batch]

        category_brand_emb = encode_texts(category_brand_texts)
        title_emb = encode_texts(title_texts)
        description_emb = encode_texts(description_texts)
        title_w_desc_emb = encode_texts(title_w_desc_texts)
        

        batch_embedding = torch.stack([category_brand_emb, title_emb, description_emb, title_w_desc_emb], dim=1)  # (batch_size, 4, hidden_dim)
        # batch_embedding = torch.stack([category_brand_emb, title_emb, description_emb], dim=1)  # (batch_size, 3, hidden_dim)
        
        embeddings.append(batch_embedding)

    embeddings = torch.cat(embeddings, dim=0).numpy()  # (num_items, 3, hidden_dim)
    print(f'Final embeddings shape: {embeddings.shape}')

    file_path = os.path.join(args.root, f"{args.dataset}.all-emb-{args.plm_name}-4-level.npy")
    np.save(file_path, embeddings)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beauty', help='Dataset name')
    parser.add_argument('--root', type=str, default="data")
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--plm_name', type=str, default='llama')
    parser.add_argument('--plm_checkpoint', type=str, default='meta-llama/Llama-3.2-3B')  # 可更改为所需的预训练模型
    parser.add_argument('--max_sent_len', type=int, default=512)
    return parser.parse_args()

def concat_embeddings(args):
    td_emb_path = os.path.join(args.root, f"{args.dataset}.emb-{args.plm_name}-td.npy")
    all_emb_path = os.path.join(args.root, f"{args.dataset}.all-emb-{args.plm_name}.npy")

    td_embeddings = np.load(td_emb_path)
    all_embeddings = np.load(all_emb_path)

    print(f"TD Embeddings shape: {td_embeddings.shape}")
    print(f"All Embeddings shape: {all_embeddings.shape}")
        
    if td_embeddings.shape[0] != all_embeddings.shape[0] or td_embeddings.shape[-1] != all_embeddings.shape[-1]:
        raise ValueError("The dimensions of TD embeddings and All embeddings do not match.")
    
    td_embeddings = np.expand_dims(td_embeddings, axis=1)
    new_all_emb = np.concatenate((td_embeddings, all_embeddings), axis=1)
    print(f"New All Embeddings shape: {new_all_emb.shape}")

    new_td_emb_path = os.path.join(args.root, f"{args.dataset}.emb-{args.plm_name}-td-updated.npy") #llama version (9922, 4, 3072)
    np.save(new_td_emb_path, new_all_emb)
    print(f"Updated TD Embeddings saved to {new_td_emb_path}")


if __name__ == '__main__':
    args = parse_args()
    args.root = os.path.join(args.root, args.dataset)

    device = set_device(args.gpu_id)
    args.device = device
    
    # concat_embeddings(args)

    item2feature = load_data(args)
    item_data = generate_text(item2feature)
    # item_data = generate_text_abaltion(item2feature)
    
    plm_tokenizer, plm_model = load_plm(args.plm_checkpoint)
    plm_model = plm_model.to(device)

    generate_embedding(args, item_data, plm_tokenizer, plm_model)
