# Score-Based Diffusion Tutorial

A comprehensive tutorial on score-based diffusion models and denoising score matching, implemented in Julia. This tutorial demonstrates how to learn score functions (gradients of log probability densities) from data using neural networks.

## Overview

Score-based diffusion models are a powerful class of generative models that learn to model data distributions by learning the score function, defined as the gradient of the log probability density: `∇_x log p(x)`. This tutorial covers:

- **Score Matching**: Learning score functions from data
- **Denoising Score Matching**: A practical variant that avoids computing intractable normalization constants
- **1D and 2D Examples**: Demonstrations on both one-dimensional and two-dimensional data distributions
- **Neural Network Training**: Using simple feedforward networks to approximate score functions

## Key Concepts

### Score Function

The score function of a probability distribution `p(x)` is defined as:

```
s(x) = ∇_x log p(x)
```

The score function points in the direction of increasing probability density and is crucial for sampling from complex distributions.

### Denoising Score Matching

Denoising score matching is a practical approach to learning score functions. Instead of directly matching the score, we:

1. Add Gaussian noise to data points: `x̃ = x + σ·z` where `z ~ N(0, I)`
2. Train a network to predict: `s_θ(x̃) ≈ -z/σ`
3. This is equivalent to matching the score of the noise-perturbed distribution

The loss function is:

```
L(θ) = E_{x~p_data, z~N(0,I)} [||s_θ(x + σ·z) + z/σ||²]
```

## Files Structure

### Main Scripts

- **`fitting_one_dimensional_scores.jl`**: Demonstrates score matching on 1D Gaussian mixture data
- **`fitting_two_dimensional_scores.jl`**: Extends the approach to 2D data with a potential energy function
- **`fitting_one_dimensional_scores_lux.jl`**: 1D implementation using Lux.jl for neural network construction
- **`fitting_two_dimensional_scores_lux.jl`**: 2D implementation using Lux.jl
- **`fitting_two_dimensional_scores.ipynb`**: Jupyter notebook version of the 2D tutorial with detailed explanations

### Supporting Files

- **`generate_potential_data.jl`**: Generates synthetic data using stochastic differential equations (SDEs) with potential energy functions
  - Creates `potential_data_1D.hdf5` and `potential_data_2D.hdf5`
  - Uses Runge-Kutta 4th order integration to simulate SDEs
- **`simple_networks.jl`**: Defines simple neural network architectures:
  - `OneLayerNetwork`: Basic single hidden layer network
  - `OneLayerNetworkWithLinearByPass`: Network with skip connection
  - `MultiLayerNetwork`: Multi-layer network
  - Includes forward pass, parameter management, and gradient computation utilities
- **`src/ScoreModeling.jl`**: Module structure (currently minimal)

## Dependencies

This tutorial requires the following Julia packages (see `Project.toml`):

- **Enzyme**: Automatic differentiation for gradient computation
- **GLMakie**: High-performance plotting and visualization
- **HDF5**: Reading/writing HDF5 data files
- **IJulia**: Jupyter notebook support
- **Lux**: Modern neural network library (for Lux-based scripts)
- **Optimisers**: Optimization algorithms (Adam, etc.)
- **ProgressBars**: Progress indicators for long-running computations
- **Zygote**: Alternative automatic differentiation backend

Standard libraries:
- `LinearAlgebra`, `Random`, `Statistics`

## Installation

1. **Install Julia** (version 1.9+ recommended)

2. **Activate the project environment**:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

This will install all required dependencies listed in `Project.toml`.

## Usage

### Generating Data

First, generate the synthetic data used in the tutorials:

```julia
julia generate_potential_data.jl
```

This creates:
- `potential_data_1D.hdf5`: 1D data from a double-well potential
- `potential_data_2D.hdf5`: 2D data from a coupled potential energy function

### Running 1D Tutorial

```julia
julia fitting_one_dimensional_scores.jl
```

This script:
1. Loads or generates 1D Gaussian mixture data
2. Defines a simple neural network
3. Trains the network using denoising score matching
4. Visualizes the learned score function vs. the exact score

### Running 2D Tutorial

```julia
julia fitting_two_dimensional_scores.jl
```

This script:
1. Loads 2D data from `potential_data_2D.hdf5`
2. Trains a network to learn the 2D score function
3. Visualizes:
   - Data distribution histograms
   - Learned score field as vector plots
   - Training/test loss curves
   - Comparison with exact score function

### Using Jupyter Notebook

For an interactive experience with detailed explanations:

```julia
using IJulia
notebook()
```

Then open `fitting_two_dimensional_scores.ipynb` in your browser.

## Key Algorithms

### Training Loop

The training process follows these steps:

1. **Data Preparation**: Load or generate data samples
2. **Network Initialization**: Create a neural network with random weights
3. **Batch Training**: For each epoch:
   - Sample batches of data points
   - Add Gaussian noise: `x̃ = x + σ·z`
   - Compute loss: `||s_θ(x̃) + z/σ||²`
   - Backpropagate gradients using Enzyme
   - Update weights using Adam optimizer
4. **Evaluation**: Compare learned score with exact score function

### Loss Functions

Two loss functions are implemented:

1. **Gaussian Mixture Loss**: Directly matches the score of a Gaussian mixture approximation
2. **Denoising Loss**: Matches the denoising objective (recommended)

The denoising loss is more stable and avoids computing expensive normalization constants.

## Visualizations

The tutorials generate several visualizations:

- **Score Function Plots**: Comparison of learned vs. exact score functions
- **Loss Curves**: Training and test loss over epochs
- **Data Distributions**: Histograms of the training data
- **Vector Fields** (2D): Visualization of the learned score function as a vector field
- **Potential Energy** (2D): Heatmap showing the underlying potential energy function

## Mathematical Background

### Potential Energy Functions

The 1D potential is:
```
V(x) = (x² - 1)² / 4
```

The 2D potential is:
```
V(x₁, x₂) = (x₁² - 1)²/4 + (x₂² - 1)²/4 + x₁·x₂/3
```

Data is generated by simulating the SDE:
```
dx = -∇V(x) dt + ε dW
```

where `dW` is a Wiener process. The stationary distribution is `p(x) ∝ exp(-V(x))`, so the score function is `s(x) = -∇V(x)`.

## Extending the Tutorial

### Adding New Architectures

Modify `simple_networks.jl` to add new network architectures. Each network type should implement:
- Forward pass (`predict` function)
- Parameter extraction (`parameters` function)
- Parameter setting (`set_parameters!` function)
- Gradient zeroing (`zero!` function)

### Trying Different Data

Modify `generate_potential_data.jl` to create data from different potential energy functions. Ensure the exact score function is updated accordingly.

### Using Lux.jl

The `*_lux.jl` scripts demonstrate how to use Lux.jl for more modern neural network construction. Lux provides:
- More flexible architecture definitions
- Better integration with optimization libraries
- Support for more complex architectures

## References

- **Score Matching**: Hyvärinen, A. (2005). "Estimation of non-normalized statistical models by score matching." *Journal of Machine Learning Research*.
- **Denoising Score Matching**: Vincent, P. (2011). "A connection between score matching and denoising autoencoders." *Neural Computation*.
- **Score-Based Generative Models**: Song, Y., et al. (2021). "Score-based generative modeling through stochastic differential equations." *ICLR*.

## Author

Andre Souza (andrenogueirasouza@gmail.com)

## License

See LICENSE file in the parent directory.

