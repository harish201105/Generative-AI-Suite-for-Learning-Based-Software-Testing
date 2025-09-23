from typing import Dict, Any
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Add parent directory to path for imports
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.json_parser import JSONResponseParser

class GeminiModel:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
            
        self.model_name = "gemini-2.0-flash-exp"
        self.temperature = 0.7
        
        # Configure Google GenAI with API key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

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
            prompt = f"{self.get_test_generation_prompt()}\n\nGenerate test cases for this code:\n{code}"

            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                }
            )

            raw_response = response.text
            
            # Use the centralized JSON parser
            result = JSONResponseParser.parse_model_response(raw_response)
            result["model"] = "Gemini 2.5 Flash"
            
            return result
            
        except Exception as e:
            return {
                "error": f"Gemini 2.5 Flash error: {str(e)}",
                "model": "Gemini 2.5 Flash"
            }
