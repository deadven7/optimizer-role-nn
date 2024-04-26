# Evaluation of Optimizers on LeNet-5 Using MNIST Data

## Project Overview
This project evaluates the performance of various optimizers, specifically Stochastic Gradient Descent (SGD), AdaGrad, and RMSprop, on the LeNet-5 convolutional neural network architecture using the MNIST dataset. The project aims to determine which optimizer yields the best accuracy and efficiency during the training process.

## Objective
To compare the effectiveness of different optimizers in training deep learning models on standard datasets, providing insights into their operational dynamics and practical performance outcomes.

## Methodology
### Data Preparation
- **Dataset Used**: MNIST dataset, consisting of 70,000 grayscale images of handwritten digits.
- **Preprocessing**: Images are normalized and centered to fit the input requirements of the LeNet-5 network.

### Model Architecture
- **LeNet-5 Network**: Consists of two convolutional layers followed by two fully connected layers and a softmax output layer.

### Optimizers Evaluated
- **SGD (Stochastic Gradient Descent)**: Updates parameters using the gradient of the loss function with respect to the model’s parameters.
- **AdaGrad (Adaptive Gradient)**: Adjusts the step size for each parameter based on historical gradients, aiming for faster convergence on sparse data.
- **RMSprop (Root Mean Square Propagation)**: Modifies the learning rate for each parameter based on recent gradients, designed to mitigate the vanishing or exploding gradient issues in SGD.

### Training and Testing
- **Training Procedure**: Each optimizer is used to train the LeNet-5 model separately, recording accuracy and training time.
- **Testing**: The trained models are evaluated against the test set to measure accuracy and generalization capabilities.

## Results
- **Performance Metrics**: The results detail the accuracy and training time for each optimizer, highlighting their strengths and weaknesses in different aspects of the training cycle.
- **Comparative Analysis**: A comparative study illustrates how each optimizer affects the learning trajectory and convergence speed.

## Conclusion
The project concludes with a recommendation on which optimizer provides the best balance between training speed and model accuracy based on the experimental results. AdaGrad emerged as particularly effective in this study, offering a robust option for training deep learning models on the MNIST dataset.

## Future Work
Further research could explore more advanced optimizers like Adam and Nadam, which combine the benefits of SGD, AdaGrad, and RMSprop, potentially leading to improved performance on more complex datasets.

## References
- Comprehensive references to all utilized resources, tools, and theoretical frameworks are included to ensure reproducibility and integrity of the project outcomes.
