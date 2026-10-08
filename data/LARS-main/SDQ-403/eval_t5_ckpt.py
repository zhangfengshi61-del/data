import os, sys, json, argparse, torch
from pathlib import Path

_real_argv = list(sys.argv)
sys.argv = ["train_t5_fp32.py"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import freerec
from train_t5_fp32 import TIGERT5, CoachForTIGER, cfg as t5cfg

ap = argparse.ArgumentParser()
ap.add_argument("--best-pt", required=True)
ap.add_argument("--sid-vocab-file", required=True)
ap.add_argument("--dataset", default="Amazon2014Sports_550_LOU")
ap.add_argument("--root", default=str(Path(__file__).resolve().parent / "data"))
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--description", default="EVAL-CKPT")
a = ap.parse_args(_real_argv[1:])

t5cfg.description = a.description
t5cfg.dataset = a.dataset
t5cfg.root = a.root
t5cfg.device = a.device
t5cfg.sid_vocab_file = a.sid_vocab_file
t5cfg.num_beams = 30
t5cfg.eval_batch_size = 96
t5cfg.apply_constrained_beam_search = False
t5cfg.prefix_loss_weight = 0.0
t5cfg.maxlen = 20
t5cfg.monitors = ["LOSS", "HitRate@1", "HitRate@5", "HitRate@10", "HitRate@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR@5", "MRR@10", "MRR@20"]
t5cfg.which4best = "NDCG@10"
t5cfg.LOG_PATH = os.path.join("logs", a.description, a.dataset, "t5")
os.makedirs(t5cfg.LOG_PATH, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = str(a.device)
import torch

dataset = getattr(freerec.data.datasets, t5cfg.dataset)(root=t5cfg.root)
model = TIGERT5(dataset)
model.load_state_dict(torch.load(a.best_pt, map_location="cpu"))
model.eval()

trainpipe = model.sure_trainpipe(t5cfg.maxlen, t5cfg.batch_size)
validpipe = model.sure_validpipe(t5cfg.maxlen, ranking=t5cfg.ranking, batch_size=t5cfg.eval_batch_size)
testpipe = model.sure_testpipe(t5cfg.maxlen, ranking=t5cfg.ranking, batch_size=t5cfg.eval_batch_size)

coach = CoachForTIGER(
    dataset=dataset, trainpipe=trainpipe, validpipe=validpipe, testpipe=testpipe,
    model=model, cfg=t5cfg,
)
print("=== VALID ===", flush=True)
coach.valid(0)
print("=== TEST ===", flush=True)
coach.test(0)
print("DONE", flush=True)
