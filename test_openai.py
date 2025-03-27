#!/usr/bin/env python3
"""
Simple test script to check OpenAI configuration
"""

import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Print OpenAI version
print(f"OpenAI version: {openai.__version__}")

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"API key is set (starts with: {api_key[:5]}...)")
else:
    print("API key is NOT set")

# Try to initialize the client
try:
    client = openai.OpenAI(api_key=api_key)
    print("Successfully initialized OpenAI client")
    
    # Test a simple completion
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            max_tokens=5
        )
        print(f"API test successful! Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"API test failed: {str(e)}")
        
except Exception as e:
    print(f"Failed to initialize OpenAI client: {str(e)}") 