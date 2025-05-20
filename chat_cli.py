#!/usr/bin/env python3
import subprocess
import sys
import os
import time

MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
LLAMA_CLI = "build/bin/llama-cli"

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def print_header():
    clear_screen()
    print("=" * 80)
    print("BitNet Chat Interface - Microsoft BitNet-b1.58-2B-4T (1-bit Neural Network)")
    print("=" * 80)
    print("Type 'exit' to quit the chat.")
    print("-" * 80)

def run_inference(prompt, max_tokens=256):
    # Create a temporary file for the prompt
    with open("temp_prompt.txt", "w") as f:
        f.write(prompt)
    
    cmd = [
        LLAMA_CLI, 
        "-m", MODEL_PATH,
        "-f", "temp_prompt.txt",
        "-n", str(max_tokens),
        "--temp", "0.7",
        "--top-p", "0.95",
        "--repeat-penalty", "1.1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Clean up the temporary file
        if os.path.exists("temp_prompt.txt"):
            os.remove("temp_prompt.txt")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")
        print(f"Error output: {e.stderr}")
        # Clean up the temporary file
        if os.path.exists("temp_prompt.txt"):
            os.remove("temp_prompt.txt")
        return None

def main():
    print_header()
    
    # Initialize chat history
    chat_history = "System: You are a helpful assistant\n\n"
    
    # Print initial bot message
    print("BitNet: Hello! I'm BitNet, a 1-bit neural network model. How can I help you today?")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("\nThank you for chatting with BitNet. Goodbye!")
            break
        
        # Update chat history
        chat_history += f"User: {user_input}\n\nAssistant: "
        
        # Run inference
        print("\nBitNet is thinking...")
        response = run_inference(chat_history)
        
        if response:
            # Extract the model's response
            try:
                # Find where the assistant's response starts
                assistant_start = response.find("Assistant: ")
                if assistant_start != -1:
                    # Extract just the assistant's response
                    assistant_response = response[assistant_start + len("Assistant: "):].strip()
                    # Update chat history with the response
                    chat_history += f"{assistant_response}\n\n"
                    # Print the response
                    print(f"\nBitNet: {assistant_response}")
                else:
                    print("\nBitNet: I couldn't generate a proper response.")
            except Exception as e:
                print(f"\nError processing response: {e}")
                print(f"Raw response: {response}")
        else:
            print("\nBitNet: Sorry, I couldn't generate a response.")

if __name__ == "__main__":
    main()