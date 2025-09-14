#!/usr/bin/env python3
"""
Flask backend for MSRA C Standard Checker
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import json
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model_and_tokenizer():
    """Load the fine-tuned Llama3 model and tokenizer"""
    global model, tokenizer
    
    try:
        print("Loading model and tokenizer...")
        
        # Configuration
        base_model_path = r"C:\Users\VICTUS\Meta-Llama-3-8B"
        lora_adapter_path = "./llama3-lora-msra/lora_adapter"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(
            model,
            lora_adapter_path,
            is_trainable=False
        )
        
        # Set eval mode for faster inference
        model.eval()
        
        print("Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def generate_response(instruction, input_code, max_length=512):
    """Generate response using the fine-tuned model"""
    
    try:
        # Create prompt
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_code}

### Response:
"""
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        generated_text = response[len(prompt):]
        
        return generated_text.strip()
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def parse_model_response(response):
    """Parse the model response to extract compliance status and rule"""
    
    try:
        # Default values
        is_compliant = False
        rule = "Unknown"
        confidence = 0.8
        
        # Check if response contains compliance indicators
        if response:
            response_lower = response.lower()
            
            # Check for compliance
            if "uygun" in response_lower and "değil" not in response_lower:
                is_compliant = True
            
            # Extract rule number
            import re
            rule_match = re.search(r'rule\s+(\d+\.?\d*)', response_lower, re.IGNORECASE)
            if rule_match:
                rule = f"Rule {rule_match.group(1)}"
            
            # Determine confidence based on response clarity
            if "uygun" in response_lower or "değil" in response_lower:
                confidence = 0.85
            else:
                confidence = 0.7
        
        return {
            "isCompliant": is_compliant,
            "message": response or "Model yanıt veremedi.",
            "rule": rule,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {
            "isCompliant": False,
            "message": "Yanıt işlenirken hata oluştu.",
            "rule": "Unknown",
            "confidence": 0.5
        }

@app.route('/api/check-code', methods=['POST'])
def check_code():
    """API endpoint to check C code against MSRA standard"""
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'code' not in data:
            return jsonify({"error": "C kodu gerekli"}), 400
        
        code = data['code'].strip()
        
        if not code:
            return jsonify({"error": "C kodu boş olamaz"}), 400
        
        # Check if model is loaded
        if model is None or tokenizer is None:
            return jsonify({"error": "Model yüklenmedi"}), 500
        
        # Standard instruction
        instruction = "Aşağıdaki C kodu MSRA C standardındaki ilgili kurala uygun mu? Kısa ve net yanıt ver."
        
        # Generate response
        print(f"Checking code: {code[:100]}...")
        start_time = time.time()
        
        response = generate_response(instruction, code)
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        if response is None:
            return jsonify({"error": "Model yanıt üretemedi"}), 500
        
        # Parse response
        result = parse_model_response(response)
        
        # Add metadata
        result["generationTime"] = round(generation_time, 2)
        result["modelUsed"] = "Fine-tuned Llama3 LoRA"
        
        return jsonify(result)
        
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({"error": f"Sunucu hatası: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    })

@app.route('/', methods=['GET'])
def index():
    """Serve the main page"""
    return """
    <html>
        <head>
            <title>MSRA C Standard Checker API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }
                code { background: #e0e0e0; padding: 2px 6px; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h1>🔍 MSRA C Standard Checker API</h1>
            <p>Fine-tuned Llama3 model ile C kodu kontrol API'si</p>
            
            <div class="endpoint">
                <h3>📋 Kod Kontrol Et</h3>
                <p><strong>Endpoint:</strong> <code>POST /api/check-code</code></p>
                <p><strong>Request Body:</strong></p>
                <pre><code>{
    "code": "int main() { return 0; }"
}</code></pre>
            </div>
            
            <div class="endpoint">
                <h3>💚 Sağlık Kontrolü</h3>
                <p><strong>Endpoint:</strong> <code>GET /api/health</code></p>
            </div>
            
            <p><em>Frontend için <code>index.html</code> dosyasını tarayıcıda açın.</em></p>
        </body>
    </html>
    """

if __name__ == '__main__':
    # Load model on startup
    if load_model_and_tokenizer():
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load model. Server not started.")
