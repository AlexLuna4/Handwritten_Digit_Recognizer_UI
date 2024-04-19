import gradio as gr
import torch
from torch import nn
from torchvision import transforms

# Load the model cause it is necessary for pytorch 

class MyMnist_ModelV0(nn.Module):
  def __init__(self,  input_shape: int, hidden_units: int, hidden_units2: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units),  # in_features = number of features in a data sample (784 pixels)
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units2, out_features=output_shape)
        )

  def forward(self, x):
      return self.layer_stack(x)

# instance of the model
load_model = MyMnist_ModelV0(input_shape=784,
                          hidden_units=256,
                          hidden_units2=128,
                          output_shape=10
)

PATH = "C:/Users/Hp Pavilion/Downloads/state_dict_model.pth" # PATH where you load the model trained

load_model.load_state_dict(torch.load(PATH))
load_model.eval()
def recognize_digit(image):
    if image is not None:
        # Preprocess of the image
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])
        image = transform(image)
        with torch.inference_mode(): # inference mode of pytoroch
            prediction = load_model(image)
        prediction = torch.softmax(prediction, dim=1)
        return {str(i): float(prediction[0][i]) for i in range(10)}
    else:
        return ""
    

demo = gr.Interface(fn=recognize_digit, 
                    inputs=gr.Image(shape=(28,28), image_mode="L", invert_colors=True, source="canvas"), 
                    outputs=gr.Label(num_top_classes=1))
demo.launch(True)
