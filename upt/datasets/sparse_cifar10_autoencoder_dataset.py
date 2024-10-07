import einops
import torch
from torchvision.datasets import CIFAR10


class SparseCIFAR10AutoencoderDataset(CIFAR10):
    def __init__(
            self,
            # how many input pixels to sample (<= 1024)
            num_inputs,
            # how many output pixels to sample (<= 1024)
            num_outputs,
            # CIFAR10 properties
            root,
            train=True,
            transform=None,
            download=False,
    ):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )
        assert num_inputs <= 1024, "CIFAR10 only has 1024 pixels, use less or equal 1024 num_inputs"
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        # CIFAR has 32x32 pixels
        # output_pos will be a tensor of shape (32 * 32, 2) with and will contain x and y indices
        # output_pos[0] = [0, 0]
        # output_pos[1] = [0, 1]
        # output_pos[2] = [0, 2]
        # ...
        # output_pos[32] = [1, 0]
        # output_pos[1024] = [31, 31]
        self.output_pos = einops.rearrange(
            torch.stack(torch.meshgrid([torch.arange(32), torch.arange(32)], indexing="ij")),
            "ndim height width -> (height width) ndim",
        ).float()

    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        assert image.shape == (3, 32, 32)
        # reshape image to sparse tensor
        x = einops.rearrange(image, "dim height width -> (height width) dim")
        pos = self.output_pos.clone()

        # subsample random input pixels (locations of inputs and outputs does not have to be the same)
        if self.num_inputs < 1024:
            if self.train:
                rng = None
            else:
                rng = torch.Generator().manual_seed(idx)
            input_perm = torch.randperm(len(x), generator=rng)[:self.num_inputs]
            input_feat = x[input_perm]
            input_pos = pos[input_perm].clone()
        else:
            input_feat = x
            input_pos = pos.clone()

        # subsample random output pixels (locations of inputs and outputs does not have to be the same)
        if self.num_outputs < 1024:
            if self.train:
                rng = None
            else:
                rng = torch.Generator().manual_seed(idx + 1)
            output_perm = torch.randperm(len(x), generator=rng)[:self.num_outputs]
            target_feat = x[output_perm]
            output_pos = pos[output_perm].clone()
        else:
            target_feat = x
            output_pos = pos.clone()

        return dict(
            index=idx,
            input_feat=input_feat,
            input_pos=input_pos,
            target_feat=target_feat,
            output_pos=output_pos,
        )
