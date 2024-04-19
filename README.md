### Handwritten Digit Classifier using PyTorch

This project aims to classify handwritten digits from 0 to 9 using a deep-layer neural network implemented in PyTorch. The comprehensive documentation covers the entire algorithm development process, including dataset acquisition, model design, training, validation, and model deployment. Below is an outline of the project structure:

1. Dataset Acquisition:
The MNIST dataset, available in the torchvision module of PyTorch, was used for training and validation. Before training the model, the dataset was split into training and validation sets to evaluate the model's performance on unseen data.

2. Model Design:
Following the dataset split, a deep-layer neural network architecture was designed. The neural network includes flattening layers, fully connected layers, and ReLU activation functions, which are common components of neural networks used for image classification tasks.

3. Model Training:
The model was trained using the training dataset obtained from the PyTorch API. During training, the model iteratively adjusted its parameters using backpropagation and gradient descent optimization algorithms to minimize the loss function.

4. Model Validation:
After training, the model's performance was evaluated using the validation dataset. Evaluation metrics such as accuracy were calculated to assess the model's ability to generalize to new examples.

5. Model Deployment:
Upon successful training and validation, the trained model was deployed as a web application using Hugging Face Spaces. This deployment allows users to interact with the model through a web browser, input handwritten digits, and receive real-time predictions.

Additional Documentation:
Comprehensive documentation of each stage of the project, including dataset splitting, model design, training, validation, and deployment, is provided. This documentation ensures transparency and reproducibility, enabling other developers to understand and replicate the project's methodology and results.

![Digit_gif](assets_img/Digit_recognition.gif) 



