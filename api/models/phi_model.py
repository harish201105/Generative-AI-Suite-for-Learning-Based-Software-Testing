from typing import Dict, Any
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv

class PhiModel:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get token from environment
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not found in environment variables. Please add it to your .env file.")
            
        self.endpoint = "https://models.github.ai/inference"
        
        # Use fast Phi-4 model
        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.token),
        )
        
        self.model = "microsoft/phi-4"  # Phi-4 model
        self.temperature = 0.7
        self.top_p = 0.9
        self.max_tokens = 2000

    def analyze_code(self, code: str) -> Dict[str, Any]:
        try:
            # Use a shorter, more focused prompt for faster response
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
                    UserMessage(f"Code:\n{code}"),
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                model=self.model
            )

            raw_response = response.choices[0].message.content
            
            # Use the centralized JSON parser
            from ..utils.json_parser import JSONResponseParser
            result = JSONResponseParser.parse_model_response(raw_response)
            result["model"] = "Phi 4"  # Update model name
            
            return result
            
        except Exception as e:
            return {
                "error": f"Phi 4 error: {str(e)}",
                "model": "Phi 4"
            }
