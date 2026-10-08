#!/usr/bin/env python
import argparse,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORKER=ROOT/"run_lars_tune_one.py"
INITIAL=[
 {"name":"c0_baseline","lr":5e-4,"dropout":0.0,"weight_decay":0.0},
 {"name":"c1_lr_low","lr":1e-4,"dropout":0.0,"weight_decay":0.0},
 {"name":"c2_lr_high","lr":2e-3,"dropout":0.0,"weight_decay":0.0},
 {"name":"c3_dropout_mid","lr":5e-4,"dropout":0.1,"weight_decay":0.0},
 {"name":"c4_dropout_high","lr":5e-4,"dropout":0.3,"weight_decay":0.0},
 {"name":"c5_wd_mid","lr":5e-4,"dropout":0.0,"weight_decay":1e-3},
 {"name":"c6_wd_high","lr":5e-4,"dropout":0.0,"weight_decay":1e-2},
]
def candidate_next(rows):
    best=max(rows,key=lambda x: float(x.get("validation_best",0.0)))
    p=best["params"]
    return [
      {"name":"refine_lr","lr":(p["lr"]+5e-4)/2,"dropout":p["dropout"],"weight_decay":p["weight_decay"]},
      {"name":"refine_dropout","lr":p["lr"],"dropout":p["dropout"]/2 if p["dropout"] else 0.05,"weight_decay":p["weight_decay"]},
      {"name":"refine_wd","lr":p["lr"],"dropout":p["dropout"],"weight_decay":p["weight_decay"]/2 if p["weight_decay"] else 5e-4},
    ]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-name",default="20260927_lars_main_binary")
    ap.add_argument("--dataset",default="Beauty",choices=["Beauty","Sports","Toys"])
    ap.add_argument("--stage",type=int,default=0)
    ap.add_argument("--gpus",default="2,3")
    ap.add_argument("--candidates-json",default=None)
    ap.add_argument("--start",type=int,default=0)
    a=ap.parse_args()
    base=ROOT/"runs"/a.run_name
    base.mkdir(parents=True,exist_ok=True)
    if a.stage==0:
        cs=INITIAL
        if a.candidates_json:
            cs=json.loads(Path(a.candidates_json).read_text())
    else:
      rows=[]
      for p in base.glob("*/"+a.dataset+"/result.json"):
        rows.append(json.loads(p.read_text()))
      if not rows: raise SystemExit("stage1 requires completed stage0 result.json")
      cs=candidate_next(rows)
    cs=cs[a.start:]
    (base/f"stage{a.stage}_candidates.json").write_text(json.dumps(cs,indent=2))
    gpu=[int(x) for x in a.gpus.split(",") if x.strip()]
    q=list(cs); active={}
    while q or active:
      for g in gpu:
        if g not in active and q:
          c=q.pop(0)
          out=base/c["name"]/a.dataset; out.mkdir(parents=True,exist_ok=True)
          log=out/"worker.log"
          cmd=[os.environ.get("PY", __import__("sys").executable),"-u",str(WORKER),"--dataset",a.dataset,"--gpu",str(g),"--candidate",c["name"],"--run-name",a.run_name,"--lr",str(c["lr"]),"--dropout",str(c["dropout"]),"--weight-decay",str(c["weight_decay"]),"--lars-steps",str(c.get("lars_steps",3)),"--sparse-weight",str(c.get("sparse_weight",0.1)),"--bridge-weight",str(c.get("bridge_weight",0.1)),"--lars-warmup",str(c.get("lars_warmup",10))]
          f=log.open("w")
          active[g]=(c,subprocess.Popen(cmd,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT),f)
          print("launched",c["name"],"gpu",g,flush=True)
      for g,(c,p,f) in list(active.items()):
        rc=p.poll()
        if rc is not None:
          f.close(); print("finished",c["name"],"gpu",g,"rc",rc,flush=True); del active[g]
      time.sleep(20)
    rows=[]
    for p in base.glob("*/"+a.dataset+"/result.json"):
        rows.append(json.loads(p.read_text()))
    if rows:
      def _vbest(x):
        v=x.get("validation_best",0.0)
        return float(v.get("NDCG@10",0.0)) if isinstance(v,dict) else float(v)
      rows.sort(key=_vbest,reverse=True)
      (base/f"stage{a.stage}_ranking.json").write_text(json.dumps(rows,indent=2))
      print(json.dumps(rows,indent=2),flush=True)
if __name__=="__main__": main()

