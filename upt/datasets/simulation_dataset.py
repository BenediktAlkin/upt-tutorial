import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    def __init__(
            self,
            root,
            # how many input points to sample
            num_inputs,
            # how many output points to sample
            num_outputs,
            # train or rollout mode
            # - train: next timestep prediction
            # - rollout: return all timesteps for visualization
            mode,
    ):
        super().__init__()
        root = Path(root).expanduser()
        self.root = root
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.mode = mode
        # discover simulations
        self.case_names = list(sorted(os.listdir(root)))
        self.num_timesteps = len(
            [
                fname for fname in os.listdir(root / self.case_names[0])
                if fname.endswith("_mesh.th")
            ],
        )
        self.mean = torch.tensor([0.0152587890625, -1.7881393432617188e-06, 0.0003612041473388672])
        self.std = torch.tensor([0.0233612060546875, 0.0184173583984375, 0.0019378662109375])

    def __len__(self):
        if self.mode == "train":
            # first timestep cant be predicted
            return len(self.case_names) * (self.num_timesteps - 1)
        elif self.mode == "rollout":
            return len(self.case_names)
        else:
            raise NotImplementedError(f"invalid mode: '{self.mode}'")

    @staticmethod
    def _load_positions(case_uri):
        x = torch.load(case_uri / "x.th", weights_only=True).float()
        y = torch.load(case_uri / "y.th", weights_only=True).float()
        # x is in [-0.5, 1.0] -> rescale to [0, 300] positional embedding is designed for positive values in the 100s
        x = (x + 0.5) * 200
        # y is in [-0.5, 0.5] -> rescale to [0, 200] positional embedding is designed for positive values in the 100s
        y = (y + 0.5) * 200
        return torch.stack([x, y], dim=1)

    def __getitem__(self, idx):
        if self.mode == "train":
            # return t and t + 1
            case_idx = idx // (self.num_timesteps - 1)
            timestep = idx % (self.num_timesteps - 1)
            case_uri = self.root / self.case_names[case_idx]
            pos = self._load_positions(case_uri)
            input_pos = pos
            output_pos = pos
            input_feat = torch.load(case_uri / f"{timestep:08d}_mesh.th", weights_only=True).float().T
            output_feat = torch.load(case_uri / f"{timestep + 1:08d}_mesh.th", weights_only=True).float().T
            # subsample inputs
            if self.num_inputs != float("inf"):
                input_perm = torch.randperm(len(input_feat))[:self.num_inputs]
                input_feat = input_feat[input_perm]
                input_pos = input_pos[input_perm]
            # subsample outputs
            if self.num_outputs != float("inf"):
                output_perm = torch.randperm(len(output_feat))[:self.num_outputs]
                output_feat = output_feat[output_perm]
                output_pos = output_pos[output_perm]
            # create input dependence to make sure that encoder works by flipping the sign of input/output features
            # - if the dataset consists of only a single sample the decoder could learn it by heart
            # - if the encoder doesnt work it would not get recognized as it is not needed
            # - flipping the sign creates an input dependence (if input sign is flipped -> flip output sign)
            # - if the encoder does not work, it will learn an average of the two samples
            # - this is only relevant because this tutorial overfits on 1 trajectory for simplicity
            if torch.rand(size=(1,)) < 0.5:
                input_feat *= -1
                output_feat *= -1
        elif self.mode == "rollout":
            # return all timesteps
            timestep = 0
            case_uri = self.root / self.case_names[idx]
            pos = self._load_positions(case_uri)
            input_pos = pos
            output_pos = pos
            data = [
                torch.load(case_uri / f"{i:08d}_mesh.th", weights_only=True).float().T
                for i in range(self.num_timesteps)
            ]
            input_feat = data[0]
            output_feat = data[1:]
            # deterministically downsample (for fast evaluation during training)
            # subsample inputs
            if self.num_inputs != float("inf"):
                rng = torch.Generator().manual_seed(idx)
                input_perm = torch.randperm(len(input_feat), generator=rng)[:self.num_inputs]
                input_feat = input_feat[input_perm]
                input_pos = input_pos[input_perm]
            # subsample outputs
            if self.num_outputs != float("inf"):
                rng = torch.Generator().manual_seed(idx)
                output_perm = torch.randperm(len(output_pos), generator=rng)[:self.num_outputs]
                output_pos = output_pos[output_perm]
                for i in range(len(output_feat)):
                    output_feat[i] = output_feat[i][output_perm]
        else:
            raise NotImplementedError

        # normalize
        input_feat -= self.mean.unsqueeze(0)
        input_feat /= self.std.unsqueeze(0)
        if isinstance(output_feat, list):
            for i in range(len(output_feat)):
                output_feat[i] -= self.mean.unsqueeze(0)
                output_feat[i] /= self.std.unsqueeze(0)
        else:
            output_feat -= self.mean.unsqueeze(0)
            output_feat /= self.std.unsqueeze(0)

        return dict(
            index=idx,
            input_feat=input_feat,
            input_pos=input_pos,
            output_feat=output_feat,
            output_pos=output_pos,
            timestep=timestep,
        )
