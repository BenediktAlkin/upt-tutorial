# Resources for UPT Tutorials

[[`Project Page`](https://ml-jku.github.io/UPT)] [[`Paper (arxiv)`](https://arxiv.org/abs/2402.12365)]]

We recommend to familiarize yourself with the following papers if you haven't already:
- [Transformer](https://arxiv.org/abs/1706.03762)
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [Perceiver IO](https://arxiv.org/abs/2107.14795)


# Tutorials


The following notebooks should provide a tutorial to get familiar with universal physics transformers.

- [Preliminaries](https://github.com/BenediktAlkin/upt-tutorial/blob/main/1_preliminaries.ipynb): provides an introduction into the concepts behind UPT
  - Sparse tensors
  - Positional encoding
  - Architecture overview
- [CIFAR10 image classification](https://github.com/BenediktAlkin/upt-tutorial/blob/main/2_image_classification.ipynb): start from a basic example (regular grid input, scalar output, easy encoder, simple classification decoder)
- [CIFAR10 autoencoder](https://github.com/BenediktAlkin/upt-tutorial/blob/main/3_image_autoencoder.ipynb): introduce the perceiver decoder to query at arbitrary positions
- [SparseCIFAR10 image classification](https://github.com/BenediktAlkin/upt-tutorial/blob/main/4_pointcloud_classification.ipynb): introduce handling point clouds via sparse tensors and supernode message passing
- [SparseCIFAR10 image autoencoder](https://github.com/BenediktAlkin/upt-tutorial/blob/main/5_pointcloud_autoencoder.ipynb): combine the handling of input point clouds with decoding arbitrary many positions
- [Simple Transient Flow](https://github.com/BenediktAlkin/upt-tutorial/blob/main/6_transient_flow_cfd.ipynb): put everything together to train UPT on a single trajectory of our transient flow simulations


