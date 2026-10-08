#!/usr/bin/env python
import ast
import argparse, gzip, json, os, time
from collections import defaultdict
from pathlib import Path
import pandas as pd

def clean(v):
    if v is None: return ""
    if isinstance(v, (list,tuple)): return " ".join(str(x) for x in v)
    if isinstance(v, dict): return " ".join(f"{k}: {x}" for k,x in v.items())
    return str(v).replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()

def load_meta(path):
    out={}
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            try: z=json.loads(line)
            except Exception:
                try: z=ast.literal_eval(line)
                except Exception: continue
            asin=z.get("asin")
            if asin:
                cats=z.get("category", z.get("categories",""))
                out[asin]=(clean(z.get("title")),clean(cats),clean(z.get("brand")))
    return out

def build(category, raw_root, canonical_root, min_user=5):
    review_path=Path(raw_root)/category["reviews"]
    meta_path=Path(raw_root)/category["meta"]
    print("loading meta",category["name"],flush=True); meta=load_meta(meta_path)
    users=defaultdict(list); pair_latest={}
    count=0
    with gzip.open(review_path,"rt",encoding="utf-8") as f:
        for line in f:
            try: z=json.loads(line)
            except Exception:
                try: z=ast.literal_eval(line)
                except Exception: continue
            u=z.get("reviewerID"); a=z.get("asin")
            if not u or not a: continue
            ts=int(z.get("unixReviewTime",0) or 0)
            pair_latest[(u,a)]=(ts,count)
            count+=1
    by_user=defaultdict(list)
    for (u,a),(ts,ord_) in pair_latest.items(): by_user[u].append((ts,ord_,a))
    seqs={u:[a for _,_,a in sorted(v)] for u,v in by_user.items() if len(v)>=min_user}
    item_set={a for s in seqs.values() for a in s}
    # deterministic item order: first appearance in user sequence after users sorted
    ordered=[]; seen=set()
    for u in sorted(seqs):
        for a in seqs[u]:
            if a not in seen: ordered.append(a); seen.add(a)
    item_id={a:i+1 for i,a in enumerate(ordered)}
    out=Path(canonical_root)/category["name"]; out.mkdir(parents=True,exist_ok=True)
    with (out/"item_text.jsonl").open("w",encoding="utf-8") as f:
        for a,i in item_id.items():
            title,cats,brand=meta.get(a,("","",""))
            f.write(json.dumps({"item":i,"asin":a,"title":title,"categories":cats,"brand":brand},ensure_ascii=False)+"\n")
    rows={}
    for split in ("train","valid","test"):
        data=[]
        for uid,u in enumerate(sorted(seqs),start=1):
            s=[item_id[a] for a in seqs[u]]
            if len(s)<5: continue
            cut={"train":len(s)-2,"valid":len(s)-1,"test":len(s)}[split]
            hist=s[:cut-1]; target=s[cut-1]
            data.append({"user":uid,"history":hist,"target":target})
        rows[split]=data
        pd.DataFrame(data).to_parquet(out/(split+".parquet"),index=False)
    stats={"dataset":category["name"],"reviews_after_dedup":len(pair_latest),
           "users":len(seqs),"items":len(item_id),"sequences":len(seqs),
           "min_sequence_length":min(map(len,seqs.values())),"max_sequence_length":max(map(len,seqs.values())),
           "meta_coverage":sum(a in meta for a in item_id)/len(item_id)}
    (out/"item_stats.json").write_text(json.dumps(stats,indent=2))
    print(stats,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--raw-root",default="/data/fszhang/amazon_raw_2014"); ap.add_argument("--canonical-root",default="/data/fszhang/RecBoard-master/data/processed"); ap.add_argument("--category",required=True,choices=["Clothing","Electronics"])
    a=ap.parse_args()
    specs={"Clothing":{"name":"Clothing","reviews":"reviews_Clothing_Shoes_and_Jewelry_5.json.gz","meta":"meta_Clothing_Shoes_and_Jewelry.json.gz"},"Electronics":{"name":"Electronics","reviews":"reviews_Electronics_5.json.gz","meta":"meta_Electronics.json.gz"}}
    build(specs[a.category],a.raw_root,a.canonical_root)

