# Deep Learning from Scratch — NumPy to PyTorch CNN / VAE

A  graduate-level deep learning exercises, progressing from raw NumPy implementations to PyTorch-based CNNs and variational autoencoders.

## Highlights

- **NumPy-first fundamentals**: linear classification, gradient descent and backpropagation implemented by hand before touching a deep learning framework.
- **MNIST pipeline**: end-to-end digit classification with a custom dataset loader and a two-layer network.
- **CNN + PyTorch**: convolutional networks and the PyTorch `nn` / `optim` workflow on the same task.
- **VAE**: training a variational autoencoder and visualising generated samples.
- `dataset_loader.py` — shared MNIST loader used across the later labs.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

> The original labs shipped `mnist.pkl.gz` alongside the notebooks. It is not committed here (binary dataset); download / place it next to `dataset_loader.py` if a lab needs it.

## Layout

| File | Topic |
|------|-------|
| `lab1_numpy.ipynb` | NumPy reminder, vectorised operations, basic plots |
| `lab2_linear_classification.ipynb` | Linear classification with a softmax / logistic model, gradient descent by hand |
| `lab3_mnist.ipynb` | MNIST digit classification with a two-layer network |
| `lab4_backprop_intro.ipynb` | Backpropagation on a small computational graph (graphviz visualisation) |
| `lab5_backprop_vectors.ipynb` | Vectorised backpropagation through a fully-connected layer |
| `lab6_cnn.ipynb` | Convolutional neural network from scratch (NumPy) |
| `lab6_pytorch.ipynb` | Same CNN task reimplemented with PyTorch `nn` |
| `lab7_vae.ipynb` | Variational autoencoder in PyTorch, with generated sample grids |
| `dataset_loader.py` | Shared MNIST loader used by labs 3, 5, 6, 7 |
