# Flow-based Neural Network Models

This repository includes TensorFlow2.x & Keras implementation of the following flow-based models:

1. NICE (Dinh et al., 2014)
2. RealNVP (Dinh et al., 2016)

Following datasets was used for testing each model.

1. Moon-shaped data
2. MNIST

## Results

|    | Moon-shaped data | MNIST |
| :-----: | :-----: | :-----: |
| NICE    | ![nice_moon](https://github.com/jaekookang/flow_based_models/blob/master/result/nice_moon.gif) | ![nice_mnist](https://github.com/jaekookang/flow_based_models/blob/master/result/nice_mnist.gif) |
| RealNVP    | ![nvp_moon](https://github.com/jaekookang/flow_based_models/blob/master/result/nvp_moon.gif) | ![nvp_mnist](https://github.com/jaekookang/flow_based_models/blob/master/result/nvp_mnist.gif) |


- Forward/Inverse mapping
    - Moon-shaped data
        - ![nvp_moon_forward](https://github.com/jaekookang/flow_based_models/blob/master/result/nvp_moon_forward.png)
        - ![nvp_moon_inverse](https://github.com/jaekookang/flow_based_models/blob/master/result/nvp_moon_inverse.png)

    - MNIST
        - ![nvp_mnist](https://github.com/jaekookang/flow_based_models/blob/master/result/nvp_mnist.png)
        - ![nvp_mnist_forward](https://github.com/jaekookang/flow_based_models/blob/master/result/nvp_mnist_forward.png)
        - ![nvp_mnist_inverse](https://github.com/jaekookang/flow_based_models/blob/master/result/nvp_mnist_inverse.png)


## Conclusions
- Due to the invertibility of layers, it is easy to visualize and interpret layer-wise operations.
- Implementing flow-based models is a bit finicky because the forward/inverse mapping can be changed based on the architecture and frameworks (tensorflow, tensorflow_probability, jax, pytorch).
- The current implementations of NICE and RealNVP is not near perfect nor purely my own work.

## TODO
- [ ] Add GLOW

# Dependencies:
- python 3.6.9
- tensorflow 2.3.0
- matplotlib
- seaborn
- numpy
- sklearn

# References:
- Dinh, L., Krueger, D., & Bengio, Y. (2014). NICE: Non-linear Independent Components Estimation. ArXiv, 1–13.
- Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using Real NVP. Arxiv.

# See:
- https://github.com/ericjang/normalizing-flows-tutorial
- https://github.com/bojone/flow
- https://github.com/MokkeMeguru/glow-realnvp-tutorial