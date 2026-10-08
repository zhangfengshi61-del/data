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
cfg.add_argument("--codebook-dim", type=int, default=128, help="dimension of codebook vector")
cfg.add_argument(
    "--apply-shared-codebook",
    type=eval,
    default=False,
    help="whether sharing the codebook",
)
cfg.add_argument("--sk-epsilons", type=str, default="0.,0.,0.003", help="epsilon for sinkhorn iteration")
cfg.add_argument("--sk-iters", type=float, default=50, help="number of iterations for sinkhorn iteration")

cfg.add_argument("--hidden-dims", type=str, default="512,128", help="hidden sizes")
cfg.add_argument("--commit-weight", type=float, default=1.0, help="weight for commitment loss")
cfg.add_argument("--dropout-rate", type=float, default=0.0, help="dropout rate")

# ode solver
cfg.add_argument(
    "--solver", type=str, 
    choices=('euler', 'rk4', 'dopri5'),
    default="dopri5"
)

# graph partition
cfg.add_argument("--partition-method", type=str, default="full")
cfg.add_argument("--ngroups", type=int, default=16, help="spectral & metis")
cfg.add_argument("--ufactor", type=int, default=7999, help="metis")
cfg.add_argument("--resolution", type=float, default=1.0, help="for louvain & leiden")

cfg.add_argument("--sem-feat-file", type=str, default="sentence-t5-xl_title_categories_brand.pkl", help="file of semantic features")

cfg.set_defaults(
    description="SDQ",
    root="../../data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=200,
    batch_size=256,
    optimizer="AdamW",
    lr=5.0e-4,
    weight_decay=0.0,
    seed=1,
)
cfg.compile()

cfg.hidden_dims = list(map(int, cfg.hidden_dims.split(",")))
cfg.sk_epsilons = list(map(float, cfg.sk_epsilons.split(",")))


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

        dims = [self.Item.embeddings.weight.size(1)] + cfg.hidden_dims + [cfg.codebook_dim]
        ACT = nn.SiLU

        self.encoder = nn.Sequential()
        for l, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:]), start=1):
            self.encoder.append(nn.Dropout(cfg.dropout_rate))
            self.encoder.append(nn.Linear(input_dim, output_dim, bias=False))
            if l < len(dims) - 1:
                self.encoder.append(ACT())

        self.quantizer = StructureDiffusionQuantizer(
            self.dataset, cfg.codebook_dim,
            features=None, solver=cfg.solver,
            num_codebooks=cfg.num_codebooks,
            num_codewords=cfg.num_codewords,
            apply_shared_codebook=cfg.apply_shared_codebook,
            commit_weight=cfg.commit_weight,
            sk_iters=cfg.sk_iters,
            sk_epsilons=cfg.sk_epsilons,
        )

        self.decoder, dims = nn.Sequential(), dims[::-1]
        for l, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:]), start=1):
            self.decoder.append(nn.Dropout(cfg.dropout_rate))
            self.decoder.append(nn.Linear(input_dim, output_dim, bias=False))
            if l < len(dims) - 1:
                self.decoder.append(ACT())

        self.partition = Mope(
            dataset,
            edge_index=self.quantizer.edge_index,
            edge_weight=self.quantizer.edge_weight,
            method=cfg.partition_method,
            resolution=cfg.resolution,
            ngroups=cfg.ngroups,
            ufactor=cfg.ufactor,
        )
        self.register_buffer(
            "groups",
            torch.from_numpy(self.partition(cfg.seed)).long()
        )

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

            num_samples = min(self.Item.count, cfg.num_codewords * 5)
            items = torch.randperm(
                self.Item.count,
            )[:num_samples]
            items, _ = torch.sort(items)

            self.quantizer.reset_local_graph(items.clone().to(cfg.device))
            x = self.Item.embeddings(items).to(cfg.device)
            z = self.encode(x)

            self.quantizer(z)

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

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        items = set(chain(*data[self.ISeq]))
        items = torch.tensor(list(items), dtype=torch.long, device=self.device)
        items, _ = torch.sort(items)
        self.quantizer.reset_local_graph(items)

        x = self.Item.embeddings(items)
        z = self.encode(x)
        q, auxiliary_loss, _ = self.quantizer(z)
        x_hat = self.decode(q)

        recon_loss = self.criterion(x_hat, x) / len(items)

        return recon_loss, auxiliary_loss

    @torch.no_grad()
    def generate_sem_ids(self):
        is_training = self.training
        self.eval()
        try:
            sem_ids = torch.zeros((self.Item.count, cfg.num_codebooks), dtype=torch.long)
            for group in self.groups.unique():
                items = torch.where(self.groups == group)[0]
                items, _ = torch.sort(items.flatten())
                self.quantizer.reset_local_graph(items)

                x = self.Item.embeddings(items)
                z = self.encode(x)
                _, _, ids = self.quantizer(z)
                sem_ids[items] = ids.detach().cpu()
            return sem_ids
        finally:
            self.train(is_training)


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

        for data in self.dataloader:
            data = self.dict_to_device(data)
            recon_loss, auxiliary_loss = self.model(data)
            loss = recon_loss + auxiliary_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.monitor(
                recon_loss.item(),
                n=1,
                reduction="mean",
                mode="train",
                pool=["RECON_LOSS"],
            )
            self.monitor(
                auxiliary_loss.item(),
                n=1,
                reduction="mean",
                mode="train",
                pool=["COMMIT_LOSS"],
            )

        if epoch % self.cfg.eval_freq == 0:
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
