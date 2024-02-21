# Import necessary libraries
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import os
import json

class DQNModel:
    def __init__(self, input_shape, action_size,load_saved_model=False):
        # Initialize main parameters
        self.action_size = action_size
        
        # Paths for saving/loading the model and parameters
        self.model_path = "model_Store"
        self.params_path = "params_Store"
        
        # Minibatch size
        self.miniSize = 320
        
        # Load saved model if specified and exists, else create a new model
        if load_saved_model and os.path.exists(self.model_path):
            print("Loading saved model...")
            self.model = tf.keras.models.load_model(self.model_path)

            new_initial_learning_rate = 0.001  # Example of a new learning rate
            new_lr_schedule = ExponentialDecay(
                new_initial_learning_rate,
                decay_steps=self.miniSize,  # Example decay steps
                decay_rate=0.90,  # Example decay rate
                staircase=False)

            # Update the optimizer with the new learning rate schedule
            self.model.optimizer.learning_rate = new_lr_schedule
            # Load training parameters
            if os.path.exists(self.params_path):
                with open(self.params_path, 'r') as f:
                    params = json.load(f)
                    self.epsilon = params['epsilon']
                    self.gamma = params['gamma']
                    self.MaxScore = params['MaxScore']
                    # Add other parameters as needed
        else:
            print("Creating new model...")
            self.model = self.build_model(input_shape)
            self.epsilon = 1.0
            self.gamma = 0.9
            self.MaxScore = 0
        
        # Clone model to create a target model for stable Q-value estimation
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        
        # Experience replay memory
        self.memory = deque(maxlen=200000)

        # Exploration parameters
        self.epsilon_min = 0.001 # Minimum exploration rate
        self.epsilon_decay = 0.995 # Decay rate for exploration
        
        
        # Model analysis tools for visualizing conv layers
        self.visualization_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('conv2dLast').output
        )
        
        self.visualization_model_2 = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('cnn1').output
        )
        
        
        
    def save_model_and_parameters(self, MaxScore):
        # Save the model weights and training parameters
        self.model.save(self.model_path)
        params = {
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'MaxScore': MaxScore,
        }
        with open(self.params_path, 'w') as f:
            json.dump(params, f)
        
    def build_model(self, input_shape):
    
        
        # CNN architecture definition using Keras Sequential API
        initial_learning_rate = 0.001
        lr_schedule = ExponentialDecay(
            initial_learning_rate,
            decay_steps=self.miniSize,
            decay_rate=0.99,
            staircase=False)
            
        # CNN architecture
        model = Sequential([
            Conv2D(32, (16, 16), padding='same', input_shape=input_shape,name='cnn1'),
            ReLU(),
            MaxPooling2D(pool_size=(8, 8)),  # Adding max pooling layer

            Conv2D(32, (8, 8), padding='same'),
            ReLU(),
            MaxPooling2D(pool_size=(2, 2)),  # Adding max pooling layer

            Conv2D(64, (3, 3), padding='same',name='conv2dLast'),
            ReLU(),
            MaxPooling2D(pool_size=(2, 2)),  # Adding max pooling layer

            Flatten(),
            Dense(256, activation='relu'),
            Dense(self.action_size,activation='linear')  # Output layer
        ])
        model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mean_squared_error')
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, action, reward, next_state, done))
        
            

    def predict(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.expand_dims(state, axis=0)
        act_values = self.model(state)
        return np.argmax(act_values[0])

    def train(self, batch_size):
        #Training loop with experience replay
        for i in range(6): # Train in mini-batches
            if len(self.memory) < 1000:  # Ensure sufficient data
                return 
                
            # Extract components of experiences
            minibatch = random.sample(self.memory, self.miniSize)
            states = np.array([t[0] for t in minibatch])
            actions = np.array([t[1] for t in minibatch])
            rewards = np.array([t[2] for t in minibatch])
            next_states = np.array([t[3] for t in minibatch])
            dones = np.array([t[4] for t in minibatch])

            # Predict future rewards and update Q-values
            target_qs = self.target_model.predict(next_states)
            max_future_qs = np.amax(target_qs, axis=1)
            
            # Compute target Q values
            for j in range(self.miniSize):
                if rewards[j] < -5:
                    target_qs[j] = rewards[j] + self.gamma * -10 * (1 - dones[j])
                else:
                    target_qs[j] = rewards[j] + self.gamma * 5 * (1 - dones[j])

            # Compute Q values for current states
            current_qs = self.model.predict(states)
            for index in range(self.miniSize):
                current_qs[index][actions[index]] = target_qs[index,0]
            
            # Possible Visualization of batches and conv layers
            # self.inspect_batch(states, current_qs, rewards, actions, next_states)
            # self.visualize_first_maxpool_output(states)
            
            
            # Train the model in one batch
            print(self.model.optimizer.learning_rate(self.model.optimizer.iterations))
            self.model.fit(states, current_qs, batch_size=batch_size, epochs=1, verbose=1)

            # Adjust the exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                print(f"Epsilon: {self.epsilon}")
                
            # Manually free memory
            del minibatch, states, actions, rewards, next_states, dones, target_qs, max_future_qs, current_qs
            gc.collect()  # Call the garbage collector
            
            
    def inspect_batch(self,states, actions, rewards, actionsOld, next_states, pause_time=1.0):
        # Visualizes a batch of states along with their corresponding actions and rewards.
        
        for i in range(len(states)):
            if rewards[i] == -100:
                plt.figure(figsize=(10, 10))
                # Display the first channel of the image state
                plt.subplot(3, 2, 1)
                plt.imshow(states[i][:,:,0].astype('uint8'))
                plt.title(f"ActionOld: {actionsOld[i]}, Action: {actions[i]}, Reward: {rewards[i]}")
                # Display the second channel of the image state
                plt.subplot(3, 2, 2)
                plt.imshow(states[i][:,:,1].astype('uint8'))
                
                plt.subplot(3, 2, 3)
                plt.imshow(next_states[i][:,:,0].astype('uint8'))
                
                plt.subplot(3, 2, 4)
                plt.imshow(next_states[i][:,:,1].astype('uint8'))
                
                rgb_image1 = np.zeros((states[0].shape[0], states[0].shape[1], 3))
                rgb_image1[:, :, 0] = states[i][:, :, 0]  # R channel
                rgb_image1[:, :, 1] = states[i][:, :, 1]  # G channel
                rgb_image1[:, :, 2] = next_states[i][:, :, 1]  # G channel
                plt.subplot(3, 2, 5)
                plt.imshow(rgb_image1.astype('uint8'))
                
                
                plt.show(block=False)
                plt.pause(pause_time)
                # Allow manual control over inspection
                #key = input("Press Enter to continue to the next observation...")  # Wait for user input
                #if key.lower() == 's':
                #    return
                plt.close()
                
    def visualize_first_maxpool_output(self, state, pause_time=1.0):
        #  Visualizes the output of the first MaxPooling layer in the model for a given state.
        
        # Prepare the state for model input
        
        # Predict the layer outputs for the given state
        output = self.visualization_model.predict(state)
        output2 = self.visualization_model_2.predict(state)
        
        # Visualize selected filters from the layer outputs
        
        for i in range(len(state)):  # Iterate over all filters
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)  # Display output from the first specified layer
            plt.imshow(output[i, :, :, 0], cmap='gray')
            plt.axis('off')
            plt.subplot(1, 2, 2)  # Display output from the first specified layer
            plt.imshow(output2[i, :, :, 0], cmap='gray')
            plt.axis('off')
            
            # Manual control to move through visualizations
            plt.show(block=False)
            plt.pause(pause_time)
            #key = input("Press Enter to continue to the next observation...")  # Wait for user input
            #if key.lower() == 's':
            #    return
            plt.close()
        

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

