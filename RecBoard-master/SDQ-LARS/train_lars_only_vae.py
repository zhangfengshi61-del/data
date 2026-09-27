#!/usr/bin/env python
"""LARS-only hierarchical residual SID tokenizer.

This implementation deliberately does not import SDQ quantizers, codebooks,
Sinkhorn, ODE diffusion, or an SDQ SID vocabulary.  It learns three fresh
codebooks from semantic item features.  At every level, budget-truncated LAR
selects the active atom for the current residual; that atom index is exported
as one SID token.  The resulting vocabulary is consumed by the unchanged
TIGER T5 protocol in the runner.
"""
import argparse, json, math, os, pickle, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from lars_solver import lar_code


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_features(root, dataset):
    p = Path(root) / "Processed" / dataset / "sentence-t5-xl_title_categories_brand.pkl"
    with p.open("rb") as f: x = pickle.load(f)
    x = torch.as_tensor(x, dtype=torch.float32)
    x = x - x.mean(0, keepdim=True)
    return F.normalize(x, dim=-1), str(p)


class LARSHRQ(nn.Module):
    def __init__(self, in_dim, code_dim=128, levels=3, words=256, lars_steps=3):
        super().__init__()
        self.levels, self.words, self.code_dim, self.lars_steps = levels, words, code_dim, lars_steps
        self.encoder = nn.Sequential(nn.Linear(in_dim, 256, bias=False), nn.SiLU(), nn.Linear(256, code_dim, bias=False))
        self.decoder = nn.Sequential(nn.Linear(code_dim, 256, bias=False), nn.SiLU(), nn.Linear(256, in_dim, bias=False))
        self.codebooks = nn.Parameter(torch.randn(levels, words, code_dim) * 0.02)

    def norm_books(self):
        return F.normalize(self.codebooks, dim=-1)

    @torch.no_grad()
    def init_books(self, z):
        # Fresh LARS-only initialization: no SDQ model/checkpoint and no KMeans.
        n = z.size(0)
        for l in range(self.levels):
            idx = torch.randperm(n, device=z.device)[:self.words]
            if idx.numel() < self.words:
                idx = idx.repeat((self.words + idx.numel() - 1) // idx.numel())[:self.words]
            self.codebooks[l].copy_(F.normalize(z[idx] + 0.01 * torch.randn_like(z[idx]), dim=-1))

    def assign(self, z):
        residual = z
        qsum = torch.zeros_like(z)
        ids_all, amps_all = [], []
        books = self.norm_books()
        for l in range(self.levels):
            # LAR active-set selection is detached, then codebook directions are
            # optimized through the hard selected atoms.
            coef = lar_code(residual.detach(), books[l].detach(), self.lars_steps)
            ids = coef.abs().argmax(dim=1)
            amp = coef.gather(1, ids[:, None]).squeeze(1).clamp(-2.0, 2.0)
            q = books[l][ids] * amp[:, None]
            qsum = qsum + q
            residual = residual - q.detach()
            ids_all.append(ids); amps_all.append(amp)
        return qsum, torch.stack(ids_all, dim=1), torch.stack(amps_all, dim=1)

    def forward(self, x):
        z = F.normalize(self.encoder(x), dim=-1)
        q, ids, amps = self.assign(z)
        # Straight-through path: forward uses LARS hard residual reconstruction;
        # encoder receives a stable identity gradient and codebooks receive q grad.
        q_st = q + (z - z.detach())
        recon = self.decoder(q_st)
        recon = F.normalize(recon, dim=-1)
        recon_loss = (1.0 - (recon * x).sum(-1)).mean()
        commit_loss = (z.detach() - q).square().mean()
        diversity = x.new_zeros(())
        entropy = x.new_zeros(())
        for l in range(self.levels):
            b = self.norm_books()[l]
            gram = b @ b.t()
            off = ~torch.eye(self.words, dtype=torch.bool, device=x.device)
            diversity = diversity + F.relu(gram[off].abs() - 0.10).mean()
            p = torch.bincount(ids[:, l], minlength=self.words).float() / max(1, ids.size(0))
            entropy = entropy + (p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum()
        # LARS reconstruction is primary; diversity and entropy are mild
        # anti-collapse regularizers and do not import an SDQ objective.
        loss = recon_loss + 0.20 * commit_loss + 0.02 * diversity + 0.002 * entropy
        return loss, recon_loss.detach(), commit_loss.detach(), ids.detach()


def export_vocab(model, x, out):
    model.eval(); ids=[]
    with torch.no_grad():
        for i in range(0, len(x), 2048):
            _, qids, _ = model.assign(model.encoder(x[i:i+2048]))
            ids.append(qids.cpu())
    ids=torch.cat(ids,0)
    vocab={f"item_{i}": tuple(f"<sid_{l}_{int(ids[i,l])}>" for l in range(model.levels)) for i in range(len(ids))}
    (out/"sid_vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    flat=[tuple(row.tolist()) for row in ids]
    collision=1.0-len(set(flat))/len(flat)
    usage=[]
    for l in range(model.levels):
        usage.append(int(ids[:,l].unique().numel()))
    return {"items":len(ids),"levels":model.levels,"words":model.words,"collision_rate":collision,"unique_per_level":usage}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True); ap.add_argument('--dataset',required=True)
    ap.add_argument('--output',required=True); ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--batch-size',type=int,default=512)
    ap.add_argument('--seed',type=int,default=2025); ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--weight-decay',type=float,default=1e-5); ap.add_argument('--levels',type=int,default=3)
    ap.add_argument('--words',type=int,default=256); ap.add_argument('--code-dim',type=int,default=128)
    ap.add_argument('--lars-steps',type=int,default=3)
    args=ap.parse_args(); seed_all(args.seed); torch.set_num_threads(4)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    x, feature_path=load_features(args.root,args.dataset); x=x.to(device)
    model=LARSHRQ(x.size(1),args.code_dim,args.levels,args.words,args.lars_steps).to(device)
    with torch.no_grad(): model.init_books(model.encoder(x[:min(len(x),max(args.words*4,2048))]))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    cfg=vars(args).copy(); cfg.update({'feature_path':feature_path,'protocol':'LARS-only-HRQ','sdq_imports':False,'device':str(device)})
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    best=1e9; history=[]; n=len(x)
    for epoch in range(1,args.epochs+1):
        model.train(); order=torch.randperm(n,device=device); sums=[0.,0.,0.]; count=0
        for s in range(0,n,args.batch_size):
            b=x[order[s:s+args.batch_size]]
            loss,rec,com,ids=model(b)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            sums[0]+=float(loss)*len(b); sums[1]+=float(rec)*len(b); sums[2]+=float(com)*len(b); count+=len(b)
        recd={'epoch':epoch,'loss':sums[0]/count,'recon':sums[1]/count,'commit':sums[2]/count}
        history.append(recd); (out/'progress.json').write_text(json.dumps(recd,indent=2),encoding='utf-8')
        if recd['loss']<best:
            best=recd['loss']; torch.save(model.state_dict(),out/'best.pt')
    torch.save(model.state_dict(),out/'model.pt')
    audit=export_vocab(model,x,out)
    audit.update({'best_train_loss':best,'final':history[-1],'feature_path':feature_path,'protocol':'LARS-only-HRQ','sdq_imports':False})
    (out/'sid_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (out/'history.json').write_text(json.dumps(history),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
