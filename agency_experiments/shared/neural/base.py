"""
Base neural network classes for agency experiments.
"""

import tensorflow as tf
from abc import ABC, abstractmethod
import numpy as np


class BaseNeuralNetwork(tf.keras.Model, ABC):
    """Base class for neural networks in agency experiments."""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def call(self, inputs, training=None):
        """Forward pass of the network."""
        pass
    
    def get_weights_as_vector(self):
        """Get all network weights as a single flat vector."""
        weights = []
        for layer in self.layers:
            layer_weights = layer.get_weights()
            for w in layer_weights:
                weights.extend(w.flatten())
        return np.array(weights)
    
    def set_weights_from_vector(self, weight_vector):
        """Set network weights from a flat vector."""
        start_idx = 0
        for layer in self.layers:
            layer_weights = layer.get_weights()
            new_weights = []
            
            for w in layer_weights:
                weight_size = w.size
                weight_shape = w.shape
                
                new_w = weight_vector[start_idx:start_idx + weight_size].reshape(weight_shape)
                new_weights.append(new_w)
                start_idx += weight_size
            
            layer.set_weights(new_weights)


class MultiOutputNetwork(BaseNeuralNetwork):
    """Neural network with multiple outputs."""
    
    def __init__(self, input_size, output_sizes, hidden_sizes=[64, 32]):
        super().__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes
        
        # Shared hidden layers
        self.hidden_layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_size, activation='relu'))
            prev_size = hidden_size
        
        # Output layers
        self.output_layers = []
        for i, output_size in enumerate(output_sizes):
            self.output_layers.append(
                tf.keras.layers.Dense(output_size, name=f'output_{i}')
            )
    
    def call(self, inputs, training=None):
        """Forward pass with multiple outputs."""
        x = inputs
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        
        # Generate outputs
        outputs = []
        for output_layer in self.output_layers:
            output = output_layer(x, training=training)
            outputs.append(output)
        
        return outputs if len(outputs) > 1 else outputs[0]