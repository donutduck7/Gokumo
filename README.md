# Gomoku AI Using AlphaZero

## Overview

This project implements a simplified version of the AlphaZero algorithm tailored for Gomoku, a two-player strategy board game. The goal was to adapt the AlphaZero framework for efficient training and playing on commercial-level hardware while maintaining competitive AI performance.

## Features

- **Customizable Gomoku Environment**: Adjustable board dimensions and win conditions.
- **Self-Play Training**: AI learns through Monte Carlo Tree Search (MCTS) and a simplified PolicyValueNet.
- **Efficient AI Design**: Simplified architecture using convolutional layers for reduced computational requirements.
- **Human-AI Interface**: Play Gomoku against a pre-trained model.

## Motivation

AlphaZero’s original implementation requires significant computational resources, making it impractical for commercial hardware. This project simplifies the architecture, enabling training and experimentation on systems with limited resources, such as non-Tensor optimized CPUs and GPUs.

## Implementation Details

1. **Gomoku Environment**:
   - Customizable board settings (e.g., 6x6 board, 4 pieces in a row to win).
   - Interactive play and self-play training modes.

2. **Self-Play and Data Augmentation**:
   - Added Dirichlet noise for exploration.
   - Leveraged rotational and flip equivalences to augment the dataset, enhancing training efficiency.

3. **PolicyValueNet**:
   - Four binary feature planes to describe the board state.
   - Outputs action probabilities and state evaluation for MCTS.

4. **Model Training**:
   - Loss function combining state value prediction, action probabilities, and L2 regularization.
   - Reduced entropy as the network learns better strategies.

5. **Evaluation**:
   - Benchmarked against a pure MCTS baseline with increasing complexity (playout count).
   - Tracked loss, entropy, and win rates over various configurations.

## Experiment Setup

- **Hardware**: 
  - AMD Ryzen 5 5600 CPU, 32GB RAM, Nvidia GeForce RTX 3060 Ti GPU.
- **Software**:
  - Python 3.11.5
  - PyTorch 2.1.1
  - Numpy 1.24.3
  - CUDA 12.4

## Results and Limitations

### Results
- Competitive performance against a pure MCTS player.
- Efficient training on commercial hardware.

### Limitations
- **Game Balance**: The first player in Gomoku has an inherent advantage, not mitigated in this implementation.
- **Model Simplifications**: Using convolutional layers instead of ResNet blocks reduces performance slightly.
- **Training Optimization**: MCTS operations are CPU-bound, which could be improved with GPU acceleration.

## Future Work

- Implement constraints to balance the first-player advantage.
- Migrate MCTS training to GPU for improved efficiency.
- Explore further architectural optimizations.

## References

- [AlphaZero: Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815)
