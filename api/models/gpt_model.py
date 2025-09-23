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
        self.max_tokens = 2000

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
        # Try models in order: GPT-5 -> GPT-5-nano -> GPT-4o -> GPT-4o-mini
        for model in self.models:
            try:
                system_prompt = """You are an expert test case generator. For the provided code, generate comprehensive test cases.

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
                    top_p=self.top_p,
                    max_tokens=self.max_tokens
                )

                raw_response = response.choices[0].message.content
                
                # Use the centralized JSON parser with correct import
                from api.utils.json_parser import JSONResponseParser
                
                result = JSONResponseParser.parse_model_response(raw_response)
                
                # Check if parsing failed
                if "error" in result:
                    return result
                
                # Add model information
                if model == "openai/gpt-5":
                    result["model"] = "GPT-5"
                elif model == "openai/gpt-4o":
                    result["model"] = "GPT-4o"
                
                return result
                
            except Exception as e:
                error_str = str(e)
                # Check if it's a rate limit error and try next model
                if "RateLimitReached" in error_str or "429" in error_str:
                    print(f"{model} rate limited, trying next model...")
                    continue
                # Check if model is unavailable and try next model
                elif "unavailable_model" in error_str or "unknown_model" in error_str:
                    print(f"{model} not available, trying next model...")
                    continue
                else:
                    return {
                        "error": f"GPT model error: {error_str}",
                        "model": "GPT-5" if model == "openai/gpt-5" else ("GPT-5-nano" if model == "openai/gpt-5-nano" else ("GPT-4o" if model == "openai/gpt-4o" else "GPT-4o-mini"))
                    }
        
        # If all models fail
        return {
            "error": "All GPT models failed",
            "model": "GPT (All variants)"
        } 