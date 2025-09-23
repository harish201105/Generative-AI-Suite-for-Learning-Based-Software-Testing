from typing import Dict, Any
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv

class Grok3Model:
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
        
        self.model = "xai/grok-3"
        self.temperature = 1.0
        self.top_p = 1.0

    def analyze_code(self, code: str) -> Dict[str, Any]:
        try:
            # Use a simple prompt to avoid timeout issues
            system_prompt = """Generate test cases for the given code. Respond with ONLY valid JSON:
{
    "test_framework": "pytest",
    "test_cases": [
        {
            "name": "test_name",
            "description": "brief description",
            "input": "input_value",
            "expected_output": "expected_result",
            "test_code": "def test_name(): pass"
        }
    ]
}"""

            response = self.client.complete(
                messages=[
                    SystemMessage(system_prompt),
                    UserMessage(f"Code to test:\n{code}"),
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                model=self.model
            )

            raw_response = response.choices[0].message.content
            
            # Use the centralized JSON parser
            from ..utils.json_parser import JSONResponseParser
            result = JSONResponseParser.parse_model_response(raw_response)
            result["model"] = "Grok-3 (Original)"
            
            return result
            
        except Exception as e:
            return {
                "error": f"Grok-3 error: {str(e)}",
                "model": "Grok-3 (Original)"
            }