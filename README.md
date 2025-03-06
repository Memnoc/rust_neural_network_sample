# Study on Neural Network Implementation

## Overview

What you've implemented is a simple feed-forward neural network with one hidden layer that learns the XOR function. XOR (exclusive OR) is a logical operation that outputs true only when inputs differ - it's a classic example for neural networks because it's not linearly separable.

## Neural Network Mathematics

In simple terms:

1. **Forward propagation**: Matrix multiplications and activation functions transform inputs into predictions
2. **Back-propagation**: Calculating gradients (derivatives) to adjust weights based on prediction errors
3. **Gradient descent**: The process of incrementally updating weights to minimize errors

The matrices in the code represent the following:

- `weights1`: Connections from input layer to hidden layer (3×2 matrix)
- `biases1`: Bias terms for the hidden layer (3 values)
- `weights2`: Connections from hidden layer to output layer (1×3 matrix)
- `biases2`: Bias term for the output layer (1 value)

## Network Architecture

- **Input layer**: 2 neurons (for the two binary inputs)
- **Hidden layer**: 3 neurons (with sigmoid activation)
- **Output layer**: 1 neuron (with sigmoid activation)

## How the Neural Network Works

### 1. Initialization

```rust
pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
}
```

- The network starts with random weights and biases
- This randomness is important so the network doesn't get stuck in symmetry

### 2. Forward Pass (Prediction)

```rust
pub fn forward(&self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let hidden_input = self.weights1.dot(input) + &self.biases1;
    let hidden_output = Self::sigmoid(&hidden_input);
    let final_input = self.weights2.dot(&hidden_output) + &self.biases2;
    let final_output = Self::sigmoid(&final_input);
    (hidden_output, final_input, final_output)
}
```

- **First layer calculation**: Multiply inputs by weights, add biases, apply sigmoid
- **Second layer calculation**: Multiply hidden outputs by weights, add biases, apply sigmoid
- The sigmoid function `1.0 / (1.0 + (-x).exp())` squashes values between 0 and 1
- using the `.exp()` method, calculates `e^x` (e raised to the power of x). The expression `(- x).exp()` calculates `e^(-x)`, which is exactly what the sigmoid function requires.

### 3. Back-propagation (Training)

```rust
pub fn train(&mut self, input: &Array1<f64>, target: &Array1<f64>, learning_rate: f64) {
    let (hidden_output, final_input, final_output) = self.forward(input);

    let output_errors = target - &final_output;
    let output_delta = &output_errors * &Self::sigmoid_derivative(&final_output);
    let hidden_errors = self.weights2.t().dot(&output_delta);
    let hidden_delta = &hidden_errors * &Self::sigmoid_derivative(&hidden_output);
}
```

## Understanding the Training Process

The training process follows these steps:

1. **Calculate output error**: The difference between target and actual output -> how off the output is. Very far off, presumably, on the first cycles.
2. **Calculate output delta**: Error multiplied by sigmoid derivative (sensitivity)
3. **Propagate error backward**: Calculate hidden layer errors by transposing weights
4. **Calculate hidden delta**: Hidden errors multiplied by sigmoid derivative
5. **Update weights and biases**: Adjust parameters using deltas, inputs, and learning rate

   The main training loop runs for 10,000 epochs, and in each epoch it processes all 4 XOR patterns:

```rust
for _ in 0..epochs {
    for (input, target) in inputs.iter().zip(targets.iter()) {
        network.train(input, target, learning_rate);
    }
}
```

For the XOR function, the inputs and expected outputs are:

- [0,0] → [0] (False XOR False = False)
- [0,1] → [1] (False XOR True = True)
- [1,0] → [1] (True XOR False = True)
- [1,1] → [0] (True XOR True = False)

## Reading the Output

After training, we `network.forward()` on each input pattern and prints the results:

```rust
for input in inputs.iter() {
    let (_, _, output) = network.forward(input);
    println!("{:?} -> {:?}", input, output);
}
```

The output should look something like:

```
[0.0, 0.0] -> [0.0346]
[0.0, 1.0] -> [0.9721]
[1.0, 0.0] -> [0.9631]
[1.0, 1.0] -> [0.0412]
```

### Interpreting the Results

- The sigmoid activation function outputs values between 0 and 1
- Values close to 0 represent "False"
- Values close to 1 represent "True"
- If the network is well-trained, it should expose values very close to 0 for [0,0] and [1,1] inputs, and very close to 1 for [0,1] and [1,0] inputs
- The exact values will vary due to random initialization, but the pattern should be clear

## Worthy improvements

1. **Add error tracking**: To monitor training progress, you could add code to calculate and print the average error after each epoch.
2. **Early stopping**: Stop training if error falls below a threshold to save computation.
3. **Learning rate decay**: Gradually reduce learning rate over time for more precise convergence.
4. **Weight initialization**: Use more sophisticated initialization methods like Xavier/Glorot.
5. **Batch training**: Update weights after processing multiple examples instead of one at a time.

## Credits

- Implementation original [source](https://evolveasdev.com/blogs/tutorial/building-a-neural-network-from-scratch-in-rust)
- An [explanation](https://www.analyticsvidhya.com/blog/2023/01/why-is-sigmoid-function-important-in-artificial-neural-networks/) of the Sigmoid function and its uses in feed-forwarding neural networks implementations.
