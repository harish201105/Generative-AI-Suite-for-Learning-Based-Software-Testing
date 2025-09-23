from typing import Dict, Any
import os
import sys
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.json_parser import JSONResponseParser

class BaseModel:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get token from environment
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not found in environment variables. Please add it to your .env file.")
            
        self.endpoint = "https://models.github.ai/inference"
        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.token),
        )
        
        # Initialize JSON parser
        self.json_parser = JSONResponseParser()

    def analyze_code(self, code: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement analyze_code method")
    
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
    
    def parse_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse the model response using the robust JSON parser."""
        return self.json_parser.parse_model_response(raw_response) 