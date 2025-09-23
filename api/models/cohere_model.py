from typing import Dict, Any
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Add parent directory to path for imports
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.json_parser import JSONResponseParser

class CohereModel:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get GitHub token from environment
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not found in environment variables.")
            
        self.endpoint = "https://models.github.ai/inference"
        self.model = "cohere/cohere-command-a"
        self.temperature = 0.7
        self.top_p = 1.0
        self.max_tokens = 2048

        # Initialize Azure AI Inference client
        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.token),
        )

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
            "name": "test_function_name",
            "description": "What this test verifies",
            "code": "def test_function_name():\n    # Test implementation\n    pass"
        }
    ]
}"""

    def analyze_code(self, code: str) -> Dict[str, Any]:
        try:
            system_prompt = self.get_test_generation_prompt()

            response = self.client.complete(
                messages=[
                    SystemMessage(system_prompt),
                    UserMessage(f"Generate test cases for this code:\n{code}"),
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                model=self.model
            )

            raw_response = response.choices[0].message.content
            
            # Use the centralized JSON parser
            result = JSONResponseParser.parse_model_response(raw_response)
            result["model"] = "Cohere Command A"
            
            return result
            
        except Exception as e:
            return {
                "error": f"Cohere Command A error: {str(e)}",
                "model": "Cohere Command A"
            } 