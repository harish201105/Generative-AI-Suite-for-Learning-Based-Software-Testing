from typing import Dict, Any
from openai import OpenAI
import os
from dotenv import load_dotenv

class GPTModel:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get token from environment
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not found in environment variables. Please add it to your .env file.")
            
        self.endpoint = "https://models.github.ai/inference"
        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=self.token,
        )
        
        # Use GPT-5 as primary model with GPT-4o as fallback
        self.models = ["openai/gpt-5", "openai/gpt-4o"]
        self.current_model = self.models[0]
        self.temperature = 1.0
        self.top_p = 0.95
        self.max_completion_tokens = 2000

    def get_test_generation_prompt(self) -> str:
        """Get the standardized prompt for test case generation."""
        return """You are an expert test case generator. For the provided code, generate comprehensive test cases.

IMPORTANT: Respond with ONLY a valid JSON object. No explanations, no markdown, no code blocks.

The JSON must have this exact structure:
{
    "test_framework": "pytest",
    "coverage_areas": ["unit tests", "edge cases", "error handling", "integration tests"],
    "test_cases": [
        {
            "name": "test_basic_functionality",
            "description": "Test the basic functionality of the function",
            "input": "sample_input_here",
            "expected_output": "expected_result_here",
            "test_code": "def test_basic_functionality():\\n    # test implementation here\\n    pass"
        }
    ]
}

Generate at least 5 comprehensive test cases covering:
1. Normal functionality
2. Edge cases (empty inputs, boundary values)
3. Error conditions (invalid inputs, exceptions)
4. Performance considerations
5. Integration scenarios

Each test case must have ALL required fields: name, description, input, expected_output, test_code.
The test_code should be actual executable Python test code using the specified framework."""

    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code and generate test cases, with fallback if rate limited."""
        print(f"🔄 GPT Model: Starting analysis for {len(code)} characters of code")
        
        # Try models in order: GPT-5 -> GPT-4o
        for idx, model in enumerate(self.models):
            try:
                print(f"🤖 Trying {model} (attempt {idx + 1}/{len(self.models)})")
                
                # Shorter, more focused system prompt to reduce processing time
                system_prompt = """Generate test cases in JSON format only. No explanations.

Required structure:
{
    "test_framework": "pytest",
    "coverage_areas": ["unit tests", "edge cases"],
    "test_cases": [
        {
            "name": "test_function_name",
            "description": "Brief description",
            "input": "input_data", 
            "expected_output": "expected_result",
            "test_code": "def test_function_name():\\n    assert function() == expected"
        }
    ]
}

Generate 3-5 test cases covering normal cases, edge cases, and error conditions."""

                print(f"📤 Sending request to {model}...")
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user", 
                            "content": f"Generate test cases for this code:\n{code}"
                        }
                    ],
                    model=model,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens
                )
                
                print(f"📥 Received response from {model}")
                raw_response = response.choices[0].message.content
                
                # Check for empty response
                if not raw_response or raw_response.strip() == "":
                    current_index = self.models.index(model)
                    if current_index < len(self.models) - 1:
                        print(f"⚠️ {model} returned empty response, trying next model...")
                        continue
                    else:
                        # Last model, return error
                        return {
                            "error": "Empty response from model",
                            "raw_output": "No content received",
                            "model": "GPT-5" if model == "openai/gpt-5" else "GPT-4o"
                        }
                
                print(f"🔍 Parsing response from {model}...")
                # Use the centralized JSON parser with correct import
                from api.utils.json_parser import JSONResponseParser
                
                result = JSONResponseParser.parse_model_response(raw_response)
                
                # Check if parsing failed
                if "error" in result:
                    # If parsing failed, continue to next model if available
                    current_index = self.models.index(model)
                    if current_index < len(self.models) - 1:
                        print(f"⚠️ {model} parsing failed, trying next model...")
                        continue
                    else:
                        # Last model, return the error with debug info
                        return {
                            "error": f"Empty response from model",
                            "raw_output": raw_response,
                            "model": "GPT-5" if model == "openai/gpt-5" else "GPT-4o"
                        }
                
                # Add model information
                if model == "openai/gpt-5":
                    result["model"] = "GPT-5"
                elif model == "openai/gpt-4o":
                    result["model"] = "GPT-4o"
                
                print(f"✅ {model} completed successfully")
                return result
                
            except Exception as e:
                error_str = str(e)
                print(f"❌ {model} failed: {error_str}")
                
                # Check if it's a rate limit error and try next model
                if "RateLimitReached" in error_str or "429" in error_str:
                    print(f"⏳ {model} rate limited, trying next model...")
                    continue
                # Check if model is unavailable and try next model
                elif "unavailable_model" in error_str or "unknown_model" in error_str:
                    print(f"🚫 {model} not available, trying next model...")
                    continue
                else:
                    # For other errors, try next model if available
                    current_index = self.models.index(model)
                    if current_index < len(self.models) - 1:
                        print(f"⚠️ {model} error, trying next model...")
                        continue
                    else:
                        return {
                            "error": f"GPT model error: {error_str}",
                            "model": "GPT-5" if model == "openai/gpt-5" else "GPT-4o"
                        }
        
        # If all models fail
        print("❌ All GPT models failed")
        return {
            "error": "All GPT models failed",
            "model": "GPT (All variants)"
        } 