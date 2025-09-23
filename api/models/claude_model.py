from typing import Dict, Any
from groq import Groq
import os
from dotenv import load_dotenv

class ClaudeModel:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Groq client
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "deepseek-r1-distill-llama-70b"
        self.temperature = 0.6
        self.top_p = 0.95
        self.max_tokens = 4096

    def analyze_code(self, code: str) -> Dict[str, Any]:
        try:
            # Use a focused prompt for better performance
            system_prompt = """Generate test cases for the given code. Respond with ONLY valid JSON:
{
    "test_framework": "pytest",
    "coverage_areas": ["unit tests", "edge cases", "error handling"],
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

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate test cases for this code:\n{code}"}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=self.max_tokens,
                stream=False
            )

            raw_response = response.choices[0].message.content
            
            # Use the centralized JSON parser
            from ..utils.json_parser import JSONResponseParser
            result = JSONResponseParser.parse_model_response(raw_response)
            result["model"] = "Claude Sonnet 4"
            
            return result
            
        except Exception as e:
            return {
                "error": f"Claude Sonnet 4 error: {str(e)}",
                "model": "Claude Sonnet 4"
            }