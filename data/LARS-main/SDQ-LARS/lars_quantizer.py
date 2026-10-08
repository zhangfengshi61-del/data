"""Training-only sparse dictionary learning on the unchanged SDQ hard path."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SDQ-403"))
from quantizer import StructureDiffusionQuantizer
from lars_solver import lar_code


class LARSStructureDiffusionQuantizer(StructureDiffusionQuantizer):
    def __init__(self, *args, lars_steps=3, sparse_weight=0.1,
                 bridge_weight=0.1, lars_ridge=1e-7, accept_margin=1.0,
                 gate_sparse=False, stage_weights=None, anchor_weight=0.0,
                 sequence_weight=0.0, sequence_temperature=0.05,
                 soft_bridge_weight=0.0, soft_temperature=0.07,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if sparse_weight < 0 or bridge_weight < 0:
            raise ValueError("loss weights must be nonnegative")
        # A single sparse budget is kept for backwards compatibility.  The
        # trust-region variant can use a decreasing budget over the SDQ
        # coarse-to-fine codebooks, e.g. [3, 2, 1].
        if isinstance(lars_steps, int):
            self.lars_steps = [lars_steps] * len(self.codebooks)
        else:
            self.lars_steps = list(lars_steps)
        if len(self.lars_steps) != len(self.codebooks):
            raise ValueError("lars_steps must have one entry per codebook")
        if any(int(x) < 1 for x in self.lars_steps):
            raise ValueError("all lars step budgets must be positive")
        self.lars_steps = [int(x) for x in self.lars_steps]
        if stage_weights is None:
            stage_weights = [1.0] * len(self.codebooks)
        if len(stage_weights) != len(self.codebooks):
            raise ValueError("stage_weights must have one entry per codebook")
        if any(float(x) < 0 for x in stage_weights):
            raise ValueError("stage_weights must be nonnegative")
        self.stage_weights = [float(x) for x in stage_weights]
        self.sparse_weight = sparse_weight
        self.bridge_weight = bridge_weight
        self.lars_ridge = lars_ridge
        if not 0.0 < accept_margin <= 1.0:
            raise ValueError("accept_margin must be in (0, 1]")
        self.accept_margin = accept_margin
        self.gate_sparse = gate_sparse
        self.anchor_weight = float(anchor_weight)
        if sequence_weight < 0 or sequence_temperature <= 0:
            raise ValueError("invalid sequence regularizer settings")
        self.sequence_weight = float(sequence_weight)
        self.sequence_temperature = float(sequence_temperature)
        if soft_bridge_weight < 0 or soft_temperature <= 0:
            raise ValueError("invalid soft bridge settings")
        self.soft_bridge_weight = float(soft_bridge_weight)
        self.soft_temperature = float(soft_temperature)
        self.loss_scale = 0.0  # initialization must be identical to SDQ
        self.last_diagnostics = []
        self.last_soft_assignments = []
        self.last_sparse_assignments = []
        self._anchor_ready = False

    @torch.no_grad()
    def set_codebook_anchor(self):
        """Freeze a copy of the baseline dictionary for trust-region training.

        The copy is registered as buffers so it follows device moves and is
        saved in checkpoints without becoming an optimised parameter.  This
        keeps the learned SID alphabet stable while LARS refines directions.
        """
        for l, codebook in enumerate(self.codebooks):
            name = f"_anchor_codebook_{l}"
            if name in self._buffers:
                self._buffers[name] = codebook.weight.detach().clone()
            else:
                self.register_buffer(name, codebook.weight.detach().clone())
        self._anchor_ready = True

    def anchor_loss(self):
        if not self._anchor_ready or self.anchor_weight <= 0:
            return self.codebooks[0].weight.new_zeros(())
        losses = []
        for l, codebook in enumerate(self.codebooks):
            anchor = getattr(self, f"_anchor_codebook_{l}")
            # Directional drift is what changes nearest-code assignments.  A
            # normalized penalty is therefore more useful than raw scale L2.
            current = F.normalize(codebook.weight, dim=-1)
            target = F.normalize(anchor, dim=-1)
            losses.append(F.mse_loss(current, target))
        return self.anchor_weight * torch.stack(losses).mean()

    def match(self, r, l):
        ids, hard = super().match(r, l)
        if self.training and (self.sequence_weight > 0 or self.soft_bridge_weight > 0):
            # Soft assignments are used only by the sequence-level auxiliary
            # objective.  The exported SID and the SDQ hard path remain
            # exactly the nearest-code assignments returned above.
            codebook = self.codebooks[l].weight
            distances = (r.unsqueeze(1) - codebook.unsqueeze(0)).square().sum(dim=-1)
            self.last_soft_assignments.append(
                F.softmax(-distances / self.sequence_temperature, dim=-1)
            )
        if self.training and self.loss_scale > 0 and (self.sparse_weight or self.bridge_weight):
            directions = F.normalize(self.codebooks[l].weight, dim=-1)
            # Inner step: fixed dictionary, jointly update the active coefficients.
            coefficients = lar_code(r.detach(), directions.detach(),
                                    self.lars_steps[l], self.lars_ridge)
            # Outer step: fixed coefficients, update normalized dictionary directions.
            sparse = coefficients @ directions
            if self.soft_bridge_weight > 0:
                # Keep the sparse teacher as a probability distribution. This
                # retains information from multiple LAR directions while the
                # exported SID remains a single hard code per level.
                teacher_dist = (sparse.unsqueeze(1) - directions.unsqueeze(0)).square().sum(dim=-1)
                self.last_sparse_assignments.append(
                    F.softmax(-teacher_dist / self.soft_temperature, dim=-1).detach()
                )
            sparse_errors = (sparse - r.detach()).square().sum(dim=1)
            sparse_loss = sparse_errors.mean()
            # Only the hard code receives the bridge gradient. Neither the encoder
            # nor the sparse teacher can reduce this term by chasing the hard code.
            hard_errors = (hard.detach() - r.detach()).square().sum(dim=1)
            # A short LAR path is not always more accurate than a hard centroid.
            # Transfer only when the sparse fit has lower error for this item.
            accept = (sparse_errors.detach() < self.accept_margin * hard_errors).to(r.dtype)
            # A rejected teacher must leave the complete SDQ objective untouched.
            # In particular, do not let its sparse reconstruction update the
            # dictionary while its bridge update is disabled.
            sparse_objective = sparse_errors * accept if self.gate_sparse else sparse_errors
            stage_weight = self.stage_weights[l]
            sparse_loss = sparse_objective.mean() * stage_weight
            bridge_loss = ((hard - sparse.detach()).square().sum(dim=1) * accept).mean() * stage_weight
            self._sparse_losses.append(sparse_loss)
            self._bridge_losses.append(bridge_loss)
            with torch.no_grad():
                hard_err = (hard - r).square().sum(dim=1).mean()
                energy = r.square().sum(dim=1).mean().clamp_min(1e-12)
                self.last_diagnostics.append(torch.stack([
                    sparse_loss.detach(), bridge_loss.detach(), hard_err,
                    sparse_loss.detach() / energy, bridge_loss.detach() / energy,
                    (coefficients.abs() > 1e-8).float().sum(dim=1).mean(),
                    (coefficients.abs().sum(dim=0) > 1e-8).float().mean(),
                    accept.mean(),
                ]))
        return ids, hard

    def forward(self, z):
        self._sparse_losses, self._bridge_losses = [], []
        self.last_diagnostics = []
        self.last_soft_assignments = []
        self.last_sparse_assignments = []
        q, loss, ids = super().forward(z)
        if self._sparse_losses:
            loss = loss + self.loss_scale * (
                self.sparse_weight * torch.stack(self._sparse_losses).mean()
                + self.bridge_weight * torch.stack(self._bridge_losses).mean()
            )
        if self.training and self.loss_scale > 0:
            loss = loss + self.loss_scale * self.anchor_loss()
        return q, loss, ids
