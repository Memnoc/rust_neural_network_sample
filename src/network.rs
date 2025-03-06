use ndarray::{Array1, Array2, Axis};
use rand::Rng;

pub struct Network {
    weights1: Array2<f64>,
    biases1: Array1<f64>,
    weights2: Array2<f64>,
    biases2: Array1<f64>,
}

impl Network {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::rng();
        let weights1 =
            Array2::from_shape_fn((hidden_size, input_size), |_| rng.random_range(-1.0..1.0));
        let biases1 = Array1::from_shape_fn(hidden_size, |_| rng.random_range(-1.0..1.0));
        let weights2 =
            Array2::from_shape_fn((output_size, hidden_size), |_| rng.random_range(-1.0..1.0));
        let biases2 = Array1::from_shape_fn(output_size, |_| rng.random_range(-1.0..1.0));
        Network {
            weights1,
            biases1,
            weights2,
            biases2,
        }
    }

    fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn sigmoid_derivative(x: &Array1<f64>) -> Array1<f64> {
        x * &(1.0 - x)
    }

    pub fn forward(&self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let hidden_input = self.weights1.dot(input) + &self.biases1;
        let hidden_output = Self::sigmoid(&hidden_input);
        let final_input = self.weights2.dot(&hidden_output) + &self.biases2;
        let final_output = Self::sigmoid(&final_input);
        (hidden_output, final_input, final_output)
    }

    pub fn train(&mut self, input: &Array1<f64>, target: &Array1<f64>, learning_rate: f64) {
        let (hidden_output, _final_input, final_output) = self.forward(input);
        let output_errors = target - &final_output;
        let output_delta = &output_errors * &Self::sigmoid_derivative(&final_output);
        let hidden_errors = self.weights2.t().dot(&output_delta);
        let hidden_delta = &hidden_errors * &Self::sigmoid_derivative(&hidden_output);

        self.weights2 = &self.weights2
            + &output_delta
                .clone()
                .insert_axis(Axis(1))
                .dot(&hidden_output.insert_axis(Axis(0)))
                * learning_rate;
        self.biases2 = &self.biases2 + &output_delta * learning_rate;

        self.weights1 = &self.weights1
            + &hidden_delta
                .clone()
                .insert_axis(Axis(1))
                .dot(&input.clone().insert_axis(Axis(0)))
                * learning_rate;
        self.biases1 = &self.biases1 + &hidden_delta * learning_rate;
    }
}
