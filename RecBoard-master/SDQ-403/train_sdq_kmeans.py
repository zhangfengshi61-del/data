import json
import os
from itertools import chain
from typing import Dict, Tuple, Union

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F

from converter import SemIDConverter
from partition import Mope
from quantizer import StructureDiffusionQuantizer

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--num-codebooks", type=int, default=3, help="number of codebooks")
cfg.add_argument(
    "--num-codewords",
    type=int,
    default=256,
    help="number of codewords per codebook",
)
cfg.add_argument(
    "--kmeans-init-method",
    type=str,
    choices=("random", "++"),
    default="random",
)
cfg.add_argument("--num-iters", type=int, default=100)

# ode solver
cfg.add_argument(
    "--solver", type=str, 
    choices=('euler', 'rk4', 'dopri5'),
    default="dopri5"
)

cfg.add_argument("--sem-feat-file", type=str, default="sentence-t5-xl_title_categories_brand.pkl", help="file of semantic features")

cfg.set_defaults(
    description="SDQ-KMeans",
    root="../../data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=1,
    batch_size=256,
    optimizer="AdamW",
    lr=5.0e-4,
    weight_decay=0.0,
    seed=1,
)
cfg.compile()


class SDQ(freerec.models.RecSysArch):
    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)

        self.Item.add_module(
            "embeddings",
            nn.Embedding.from_pretrained(
                self.normalize(
                    freerec.utils.import_pickle(
                        os.path.join(
                            self.dataset.path,
                            cfg.sem_feat_file,
                        )
                    )
                ),
                freeze=True,
            ),
        )
        cfg.codebook_dim = self.Item.embeddings.weight.data.size(1)

        self.encoder = nn.Identity()
        self.quantizer = StructureDiffusionQuantizer(
            self.dataset, cfg.codebook_dim,
            features=None, solver=cfg.solver,
            num_codebooks=cfg.num_codebooks,
            num_codewords=cfg.num_codewords,
            apply_shared_codebook=False,
            trainable=False,
            kmeans_init_method=cfg.kmeans_init_method,
            num_iters=cfg.num_iters
        )
        self.decoder = nn.Identity()

        self.criterion = nn.MSELoss(reduction="sum")

        self.reset_parameters()

    def normalize(self, feats: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            feats - feats.mean(dim=0, keepdim=True),
            dim=-1
        )

    @freerec.utils.timemeter
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # codebook initialization
        with torch.no_grad():
            for codebook in self.quantizer.codebooks:
                codebook.requires_kmeans_init_ = True
            
            self.encoder.to(cfg.device)
            self.quantizer.to(cfg.device)

            items = torch.arange(self.Item.count)
            self.quantizer.reset_local_graph(items.clone().to(cfg.device))
            x = self.Item.embeddings(items).to(cfg.device)
            z = self.encode(x)

            _, _, ids = self.quantizer(z)

            self.register_buffer(
                "sem_ids",
                ids.detach().clone()
            )

    def sure_trainpipe(self, batch_size: int = 512):
        return (
            self.dataset
            .train()
            .shuffled_seqs_source(maxlen=50)
            .batch_(batch_size)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)  # (B, D)
        return z

    def decode(self, q: torch.Tensor):
        x_hat = self.decoder(q)  # (B, D)
        return F.normalize(x_hat, dim=-1)  # normalization !!!

    @torch.no_grad()
    def generate_sem_ids(self):
        return self.sem_ids


class CoachForSDQ(freerec.launcher.Coach):
    @freerec.ddp.main_process_only
    def save_sid_vocab(self) -> None:
        sem_ids = self.get_res_sys_arch().generate_sem_ids()
        sid_vocab = {}
        for item_id, sids in enumerate(sem_ids.tolist()):
            sids = [SemIDConverter.SID_FORMAT.format(level=level, id=sid) for level, sid in enumerate(sids)]
            sid_vocab[SemIDConverter.format(item_id)] = tuple(sids)
        vocab_file = os.path.join(self.cfg.LOG_PATH, "sid_vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as file:
            json.dump(sid_vocab, file)

    def set_other(self):
        self.register_metric("RECON_LOSS", lambda x: x, best_caster=min)
        self.register_metric("COMMIT_LOSS", lambda x: x, best_caster=min)
        self.register_metric("PPL", lambda x: x, best_caster=max)
        self.register_metric("COLLISION_RATE", lambda x: x, best_caster=min)
        for i in range(self.cfg.num_codebooks):
            self.register_metric(f"PPL#{i}", lambda x: x, best_caster=max)

    def train_per_epoch(self, epoch: int):
        self.save_sid_vocab()

    def evaluate(self, epoch, step=-1, mode="valid"):
        sem_ids = self.get_res_sys_arch().generate_sem_ids().cpu()
        counts = torch.zeros((cfg.num_codewords, cfg.num_codebooks))
        counts.scatter_add_(0, sem_ids, torch.ones_like(sem_ids, dtype=torch.float))
        uniques = set([tuple(id_) for id_ in sem_ids.tolist()])

        freqs = counts.div(counts.sum(dim=0, keepdim=True))
        perplexity = ((freqs + 1.0e-8).log() * freqs).sum(dim=0).neg().exp().tolist()

        ppls = []
        for i, ppl in enumerate(perplexity):
            ppls.append(ppl)
            self.monitor(ppl, n=1, mode="valid", pool=[f"PPL#{i}"])

        self.monitor(
            sum(ppls),
            n=len(ppls),
            mode=mode,
            reduction="sum",
            pool=["PPL"],
        )
        self.monitor(
            (self.Item.count - len(uniques)) / self.Item.count,
            n=1,
            mode=mode,
            pool=["COLLISION_RATE"],
        )


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(
            cfg.root,
            cfg.dataset,
            tasktag=cfg.tasktag,
        )

    model = SDQ(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.batch_size)

    coach = CoachForSDQ(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=trainpipe,
        testpipe=None,
        model=model,
        cfg=cfg,
    )
    coach.fit()


if __name__ == "__main__":
    main()
