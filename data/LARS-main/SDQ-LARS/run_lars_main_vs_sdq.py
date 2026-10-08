#!/usr/bin/env python
import argparse, json, os, subprocess
from pathlib import Path
import sys
PY = os.environ.get("PY", sys.executable)
SDQ = Path(__file__).resolve().parents[1] / "SDQ-403"
LARS = Path(__file__).resolve().parents[1] / "SDQ-LARS"
DATASET_ROOT = SDQ / "data"
RUN_ROOT = LARS / "runs"
def run_stage(name, cmd, env, log, cwd):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("[start]", name, " ".join(cmd), flush=True)
    with log.open("w") as f:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"{name} failed rc={rc}; see {log}")
    print("[done]", name, flush=True)
def read_result(log_dir):
    import pickle
    p = Path(log_dir) / "data" / "best.pkl"
    with p.open("rb") as f: x = pickle.load(f)
    return {"validation_best": float(x["valid"]), "test_selected": float(x["best"]), "source": str(p)}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["Clothing","Electronics"])
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument("--run-name", default="20260927_lars_main_common")
    a = ap.parse_args()
    ds = "Amazon2014" + a.dataset + "_550_LOU"
    out = RUN_ROOT / a.run_name / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(a.gpu), "TOKENIZERS_PARALLELISM":"false", "OMP_NUM_THREADS":"4", "MKL_NUM_THREADS":"4"})
    cfg_vae = SDQ / "configs" / "sdq" / f"{ds}.yaml"
    cfg_t5 = SDQ / "configs" / "t5" / f"{ds}.yaml"
    sdq_desc = f"SDQ-{a.run_name}-{a.dataset}"
    lars_desc = f"LARS-{a.run_name}-main-{a.dataset}"
    sdq_log = SDQ / "logs" / sdq_desc / ds
    lars_log = LARS / "logs" / lars_desc / ds
    try:
        if not (sdq_log/"vae"/"sid_vocab.json").exists():
            run_stage("sdq_vae", [PY,"-u",str(SDQ/"train_sdq_vae.py"),"--config",str(cfg_vae),"--root",str(DATASET_ROOT),"--description",sdq_desc,"--id","vae","--device","0","--epochs","100","--seed","2025"], env, out/"sdq_vae.log", SDQ)
        else:
            print("[skip] sdq_vae already done", flush=True)
        sdq_t5_desc = sdq_desc + "-T5"
        sdq_t5_log = SDQ / "logs" / sdq_t5_desc / ds
        if not (sdq_t5_log/"t5"/"data"/"best.pkl").exists():
            run_stage("sdq_t5", [PY,"-u",str(SDQ/"train_t5_fp32.py"),"--config",str(cfg_t5),"--root",str(DATASET_ROOT),"--sid-vocab-file",str(sdq_log/"vae"/"sid_vocab.json"),"--description",sdq_t5_desc,"--id","t5","--device","0","--num-workers","0","--num-beams","30","--batch-size","512","--eval-batch-size","96","--epochs","200","--seed","2025","--apply-constrained-beam-search","False","--prefix-loss-weight","0.0","--log2file","--log2console"], env, out/"sdq_t5.log", SDQ)
        else:
            print("[skip] sdq_t5 already done", flush=True)
        if not (lars_log/"vae"/"sid_vocab.json").exists():
            run_stage("lars_vae", [PY,"-u",str(LARS/"train_lars_vae.py"),"--config",str(cfg_vae),"--root",str(DATASET_ROOT),"--description",lars_desc,"--id","vae","--device","0","--epochs","100","--seed","2025","--lars-steps","3","--sparse-weight","0.1","--bridge-weight","0.1","--lars-warmup","10"], env, out/"lars_vae.log", LARS)
        else:
            print("[skip] lars_vae already done", flush=True)
        lars_t5_desc = lars_desc + "-T5"
        lars_t5_log = SDQ / "logs" / lars_t5_desc / ds
        if not (lars_t5_log/"t5"/"data"/"best.pkl").exists():
            run_stage("lars_t5", [PY,"-u",str(SDQ/"train_t5_fp32.py"),"--config",str(cfg_t5),"--root",str(DATASET_ROOT),"--sid-vocab-file",str(lars_log/"vae"/"sid_vocab.json"),"--description",lars_t5_desc,"--id","t5","--device","0","--num-workers","0","--num-beams","30","--batch-size","512","--eval-batch-size","96","--epochs","200","--seed","2025","--apply-constrained-beam-search","False","--prefix-loss-weight","0.0","--log2file","--log2console"], env, out/"lars_t5.log", SDQ)
        else:
            print("[skip] lars_t5 already done", flush=True)
        sdq = read_result(sdq_t5_log); lars = read_result(lars_t5_log)
        result = {"dataset":a.dataset,"protocol":{"vae_epochs":100,"t5_epochs":200,"seed":2025,"beam":30,"fp32":True,"train_batch":512,"eval_batch":96,"apply_constrained_beam_search":False,"prefix_loss_weight":0.0},"sdq_vae_log":str(sdq_log),"sdq_t5_log":str(sdq_t5_log),"lars_vae_log":str(lars_log),"lars_t5_log":str(lars_t5_log),"sdq":sdq,"lars_main":lars,"relative_gain":(lars["test_selected"]/sdq["test_selected"]-1.0)}
        (out/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
        print(json.dumps(result,indent=2), flush=True)
    except Exception as e:
        (out/"error.json").write_text(json.dumps({"error":str(e)},indent=2))
        raise
if __name__ == "__main__":
    main()

