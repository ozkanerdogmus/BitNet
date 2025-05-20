#!/usr/bin/env python3
import os
import json
import subprocess
import tempfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Configuration
MODEL_PATH = "../models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
LLAMA_CLI = "../build/bin/llama-cli"
HOST = "0.0.0.0"
PORT = 12001

# Ensure paths are absolute
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH))
LLAMA_CLI = os.path.abspath(os.path.join(os.path.dirname(__file__), LLAMA_CLI))

class BitNetHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Set the directory to serve static files from
        self.directory = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            prompt = data.get('prompt', '')
            
            # Format the prompt for the model
            formatted_prompt = f"User: {prompt}\n\nAssistant: "
            
            # Generate response
            response = self.generate_response(formatted_prompt)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response_data = {
                'response': response
            }
            
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
        else:
            self.send_error(404)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def generate_response(self, prompt, max_tokens=256):
        # Create a temporary file for the prompt
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
            temp.write(prompt)
            temp_path = temp.name
        
        try:
            # Run the model
            cmd = [
                LLAMA_CLI,
                "-m", MODEL_PATH,
                "-f", temp_path,
                "-n", str(max_tokens),
                "--temp", "0.7",
                "--top-p", "0.95",
                "--repeat-penalty", "1.1"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Extract the model's response
            output = result.stdout
            
            # Find where the assistant's response starts
            assistant_start = output.find("Assistant: ")
            if assistant_start != -1:
                # Extract just the assistant's response
                assistant_response = output[assistant_start + len("Assistant: "):].strip()
                return assistant_response
            else:
                return "I couldn't generate a proper response."
        
        except subprocess.CalledProcessError as e:
            print(f"Error running inference: {e}")
            print(f"Error output: {e.stderr}")
            return "Sorry, there was an error processing your request."
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

def run_server():
    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, BitNetHandler)
    print(f"Starting BitNet web server on http://{HOST}:{PORT}")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()