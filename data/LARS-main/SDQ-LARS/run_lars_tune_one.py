#!/usr/bin/env python
import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY=os.environ.get("PY", sys.executable)
SDQ=ROOT/"SDQ-403"
LARS=ROOT/"SDQ-LARS"
def run(name, cmd, env, log, cwd):
    with log.open("w") as f:
        p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
        rc=p.wait()
    if rc: raise RuntimeError(f"{name} rc={rc}")
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True,choices=["Beauty","Sports","Toys"])
    ap.add_argument("--gpu",required=True,type=int)
    ap.add_argument("--candidate",required=True)
    ap.add_argument("--run-name",required=True)
    ap.add_argument("--lr",required=True,type=float)
    ap.add_argument("--dropout",required=True,type=float)
    ap.add_argument("--weight-decay",required=True,type=float)
    ap.add_argument("--vae-only",action="store_true")
    ap.add_argument("--lars-steps",type=int,default=3)
    ap.add_argument("--sparse-weight",type=float,default=0.1)
    ap.add_argument("--bridge-weight",type=float,default=0.1)
    ap.add_argument("--lars-warmup",type=int,default=10)
    a=ap.parse_args()
    ds=f"Amazon2014{a.dataset}_550_LOU"
    out=LARS/"runs"/a.run_name/a.candidate/a.dataset
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES":str(a.gpu),"TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"})
    cfg_vae=SDQ/"configs/sdq"/f"{ds}.yaml"
    cfg_t5=SDQ/"configs/t5"/f"{ds}.yaml"
    desc=f"LARS-{a.run_name}-{a.candidate}-{a.dataset}"
    v_log=LARS/"logs"/desc/ds
    t_desc=desc+"-T5"
    t_log=SDQ/"logs"/t_desc/ds
    (out/"config.json").write_text(json.dumps({"dataset":a.dataset,"candidate":a.candidate,"gpu":a.gpu,"vae_lr":a.lr,"vae_dropout":a.dropout,"vae_weight_decay":a.weight_decay,"protocol":{"vae_epochs":100,"t5_epochs":200,"seed":2025,"beam":30,"fp32":True,"train_batch":512,"eval_batch":96,"apply_constrained_beam_search":False,"t5_params_fixed":True}},indent=2))
    if (out/"result.json").exists(): return
    try:
        v_vocab=v_log/"vae/sid_vocab.json"
        lock=v_log/"vae/.vae_lock"
        if not a.vae_only and not v_vocab.exists() and lock.exists():
            t0=time.time()
            while lock.exists() and not v_vocab.exists() and time.time()-t0<3*3600:
                time.sleep(30)
        if v_vocab.exists():
            if a.vae_only: return
        else:
            vae=[PY,"-u",str(LARS/"train_lars_vae.py"),"--config",str(cfg_vae),"--root",str(SDQ/"data"),"--description",desc,"--id","vae","--device","0","--epochs","100","--seed","2025","--lars-steps",str(a.lars_steps),"--sparse-weight",str(a.sparse_weight),"--bridge-weight",str(a.bridge_weight),"--lars-warmup",str(a.lars_warmup),"--lr",str(a.lr),"--dropout-rate",str(a.dropout),"--weight-decay",str(a.weight_decay)]
            lock.parent.mkdir(parents=True,exist_ok=True)
            lock.write_text(f"{os.getpid()} {time.time()}")
            try:
                run("vae",vae,env,out/"vae.log",LARS)
            finally:
                if lock.exists(): lock.unlink()
        if a.vae_only: return
        t_lock=t_log/"t5/.t5_lock"
        if t_lock.exists(): return
        t_lock.parent.mkdir(parents=True,exist_ok=True)
        t_lock.write_text(f"{os.getpid()} {time.time()}")
        try:
            t5=[PY,"-u",str(SDQ/"train_t5_fp32.py"),"--config",str(cfg_t5),"--root",str(SDQ/"data"),"--sid-vocab-file",str(v_log/"vae/sid_vocab.json"),"--description",t_desc,"--id","t5","--device","0","--num-workers","0","--num-beams","30","--batch-size","512","--eval-batch-size","96","--epochs","200","--seed","2025","--apply-constrained-beam-search","False","--prefix-loss-weight","0.0","--log2file","--log2console"]
            run("t5",t5,env,out/"t5.log",SDQ)
            import pickle
            p=t_log/"t5/data/best.pkl"
            with p.open("rb") as f: s=pickle.load(f)
            result={"dataset":a.dataset,"candidate":a.candidate,"params":{"lr":a.lr,"dropout":a.dropout,"weight_decay":a.weight_decay},"validation_best":s.get("valid"),"test_selected":s.get("best"),"source":str(p)}
            (out/"result.json").write_text(json.dumps(result,indent=2))
        finally:
            if t_lock.exists(): t_lock.unlink()
    except Exception as e:
        (out/"error.json").write_text(json.dumps({"error":str(e)},indent=2))
        raise
if __name__=="__main__": main()

