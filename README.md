# SnakeAI Game

SnakeAI brings an innovative twist to the classic Snake game by incorporating Artificial Intelligence. This project allows players to not only enjoy the traditional snake game but also to engage with an AI that learns to play the game using computer vision. The entire game frame is utilized as input to the neural network, enabling the AI to make informed decisions based on the visual state of the game.

## Features

- **AI Training**: Utilize computer vision to train an AI model to master the Snake game through observation and learning.
- **Single Player vs. AI**: Challenge yourself against the AI and experience a game where your opponent learns from the gameplay.
- **Classic Mode**: Relive the nostalgia with a classic mode that allows for straightforward snake gameplay without AI intervention.

## Requirements

This game and its AI component are powered by a suite of robust libraries. Ensure you have the following installed:

- Python 3.x
- TensorFlow
- OpenCV-Python
- Pygame
- NumPy
- Matplotlib

## Installation

To set up your environment for the game, install the necessary libraries using pip:


```bash
pip install pygame numpy tensorflow matplotlib opencv-python
```



## Structure

- "main_AI.py": The script used to train the AI model.
- "main_Test.py": A script that allows you to test the current best model.
- "main_solo.py": A script for a classic one-player game without AI.

## Usage

### Training the AI

To train the model, run:


```bash
python main_AI.py
```



This will initiate the training process. The model will learn by playing against itself. Progress can be monitored via the console output.

### Playing Against the AI

Once the model is trained, you can test the model:


```bash
python main_Test.py
```


### Two-Player Mode

To play a game of Snake:


```bash
python main_solo.py
```



Player 1 uses the W, D, A and S keys.

## Custom Game Environment

The game leverages a custom snake environment ("snake_game.py") tailored for AI interaction and classic gameplay.

## AI Model

The AI ("model_AI.py") is built using TensorFlow and employs a Deep Q-Network (DQN) to learn the game.

## Contributing

Contributions are welcome! If you have suggestions for improving the game or the AI, feel free to fork the repository and submit a pull request.

