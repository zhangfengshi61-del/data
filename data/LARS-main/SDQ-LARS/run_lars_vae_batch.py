#!/usr/bin/env python
import argparse, json, os, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORKER=ROOT/"run_lars_tune_one.py"
PY=os.environ.get("PY", __import__("sys").executable)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidates-json",required=True)
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--run-name",required=True)
    ap.add_argument("--start",type=int,default=2)
    ap.add_argument("--slots",required=True)
    a=ap.parse_args()
    cs=json.loads(Path(a.candidates_json).read_text())[a.start:]
    kept=[]
    for c in cs:
        v_log=ROOT/"logs"/f"LARS-{a.run_name}-{c['name']}-{a.dataset}"/f"Amazon2014{a.dataset}_550_LOU"
        if (v_log/"vae/sid_vocab.json").exists() or (v_log/"vae/.vae_lock").exists():
            print("skip",c["name"],"already running/done",flush=True)
        else:
            kept.append(c)
    cs=kept
    slots=[int(x) for x in a.slots.split(",") if x.strip()]
    q=list(cs); active={}
    base=ROOT/"runs"/a.run_name
    while q or active:
        for i in range(len(slots)):
            if i not in active and q:
                c=q.pop(0)
                out=base/c["name"]/a.dataset; out.mkdir(parents=True,exist_ok=True)
                log=out/"vae_only.log"
                cmd=[PY,"-u",str(WORKER),"--dataset",a.dataset,"--gpu",str(slots[i]),"--candidate",c["name"],"--run-name",a.run_name,"--lr",str(c["lr"]),"--dropout",str(c["dropout"]),"--weight-decay",str(c["weight_decay"]),"--lars-steps",str(c.get("lars_steps",3)),"--sparse-weight",str(c.get("sparse_weight",0.1)),"--bridge-weight",str(c.get("bridge_weight",0.1)),"--lars-warmup",str(c.get("lars_warmup",10)),"--vae-only"]
                f=log.open("w")
                active[i]=(c,subprocess.Popen(cmd,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT),f)
                print("launched vae",c["name"],"gpu",slots[i],flush=True)
        for i,(c,p,f) in list(active.items()):
            rc=p.poll()
            if rc is not None:
                f.close(); print("finished vae",c["name"],"gpu",slots[i],"rc",rc,flush=True); del active[i]
        time.sleep(15)
    print("ALL VAE DONE",flush=True)
if __name__=="__main__": main()
