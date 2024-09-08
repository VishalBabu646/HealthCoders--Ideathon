import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from csv_reader import csv_prompter  # Ensure this module and function exist

# Set environment variable for CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize model, tokenizer, and generator
model = None
tokenizer = None
generator = None

def load_model(model_name, device="cuda"):
    global model, tokenizer, generator

    print("Loading " + model_name + "...")

    # Configure GPU count
    gpu_count = torch.cuda.device_count()
    print('GPU count:', gpu_count)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Adjust based on availability
        low_cpu_mem_usage=True,
        cache_dir="cache"
    ).to(device)

    # Set the generator method
    generator = model.generate

# Load model (ensure the model path or name is correct)
load_model("chatDoctor100k/")

# Initial chat message
First_chat = "ChatDoctor: I am ChatDoctor, what medical questions do you have?"
print(First_chat)

def go():
    invitation = "ChatDoctor: "
    human_invitation = "Patient: "

    # Input
    msg = input(human_invitation)
    print("")

    # Ensure csv_prompter function works with the generator and tokenizer
    response = csv_prompter(generator, tokenizer, msg)  # Ensure this function is implemented correctly

    print("")
    print(invitation + response)
    print("")

while True:
    go()
