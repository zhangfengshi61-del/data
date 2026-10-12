"""Guard beam dump candidate width and aligned per-depth scores."""
import tempfile
import unittest
from pathlib import Path

import torch
from sid_mlp.dataset import attach_beam_dump


class _Dataset:
    def __len__(self):
        return 1


class TestBeamDump(unittest.TestCase):
    @staticmethod
    def _dump(step_logp=None):
        codes = torch.tensor([[
            [10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30],
        ]], dtype=torch.uint8)
        data = {
            "codes": codes,
            "scores": torch.tensor([[.1, .9, .5]]),
            "nvalid": torch.tensor([3]),
            "W": 3,
            "meta": {"anchors": 1},
        }
        if step_logp is not None:
            data["step_logp"] = step_logp
        return data

    def _load(self, data, topm):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "beam.pt"
            torch.save(data, path)
            return attach_beam_dump(_Dataset(), str(path), torch.device("cpu"), topm)

    def test_requested_width_must_be_represented_in_dump(self):
        with self.assertRaisesRegex(ValueError, "Requested M=4"):
            self._load(self._dump(), 4)

    def test_step_scores_must_match_candidate_shape(self):
        with self.assertRaisesRegex(ValueError, "step_logp"):
            self._load(self._dump(torch.zeros(1, 2, 4)), 2)

    def test_topm_permutation_keeps_scores_and_steps_aligned(self):
        steps = torch.stack([
            torch.full((4,), -.1),
            torch.full((4,), -.2),
            torch.full((4,), -.3),
        ], 0).unsqueeze(0)
        ds = self._load(self._dump(steps), 2)
        self.assertEqual(ds.bm_codes.shape, (1, 2, 4))
        self.assertEqual(ds.bm_codes[0, :, 0].tolist(), [20, 30])
        self.assertEqual(ds.bm_scores[0].tolist(), [float(x) for x in
                         torch.tensor([.9, .5])])
        torch.testing.assert_close(ds.bm_steplp[0, :, 0], torch.tensor([-.2, -.3]))
        self.assertEqual(ds.bm_nvalid[0].item(), 2)


if __name__ == "__main__":
    unittest.main()
