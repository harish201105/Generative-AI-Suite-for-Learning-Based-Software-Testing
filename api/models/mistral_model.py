from typing import Dict, Any
import os
from mistralai import Mistral, UserMessage, SystemMessage
from dotenv import load_dotenv

# Add parent directory to path for imports
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.json_parser import JSONResponseParser

class MistralModel:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get token from environment
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not found in environment variables.")
            
        self.endpoint = "https://models.github.ai/inference"
        self.model = "mistral-ai/mistral-medium-2505"
        self.temperature = 1.0
        self.top_p = 1.0
        self.max_tokens = 1000
        
        # Initialize Mistral client
        self.client = Mistral(api_key=self.token, server_url=self.endpoint)

    def get_test_generation_prompt(self) -> str:
        """Get the standardized prompt for test case generation."""
        return """You are an expert test case generator. Generate comprehensive test cases.

IMPORTANT: Respond with ONLY a valid JSON object. No explanations, no markdown.

The JSON must have this exact structure:
{
    "test_framework": "pytest",
    "coverage_areas": ["unit tests", "edge cases", "error handling"],
    "test_cases": [
        {
            "name": "test_basic_functionality",
            "description": "Test the basic functionality",
            "input": "sample_input_here",
            "expected_output": "expected_result_here",
            "test_code": "def test_basic():\\n    pass"
        }
    ]
}

Generate at least 5 test cases covering normal functionality, edge cases, and error conditions."""

    def analyze_code(self, code: str) -> Dict[str, Any]:
        try:
            system_prompt = self.get_test_generation_prompt()

            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=f"Generate test cases for this code:\n{code}"),
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )

            raw_response = response.choices[0].message.content
            
            # Use the centralized JSON parser
            result = JSONResponseParser.parse_model_response(raw_response)
            result["model"] = "Mistral Medium 2505"
            
            return result
            
        except Exception as e:
            return {
                "error": f"Mistral Medium 2505 error: {str(e)}",
                "model": "Mistral Medium 2505"
            }
