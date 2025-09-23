from typing import Dict, Any
import os
from groq import Groq
from dotenv import load_dotenv

class LlamaModel:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get Groq API key from environment
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
            
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        
        # Model configuration for Llama-4-Maverick
        self.model = "meta-llama/llama-4-maverick-17b-128e-instruct"
        self.temperature = 0.3  # Focused responses
        self.max_completion_tokens = 1200  # Reasonable limit
        self.top_p = 0.85  # Good balance
        
    def get_fast_config(self):
        """Get configuration optimized for speed."""
        return {
            'temperature': 0.1,  # Very focused
            'top_p': 0.7,        # More focused
            'max_completion_tokens': 800    # Even shorter for speed
        }

    def get_test_generation_prompt(self) -> str:
        """Get the standardized prompt for test case generation."""
        return """Generate test cases in JSON format only. Be concise.

Required JSON structure:
{
    "test_framework": "pytest",
    "coverage_areas": ["unit tests", "edge cases", "error handling"],
    "test_cases": [
        {
            "name": "test_function_name",
            "description": "Brief test description",
            "code": "def test_function_name():\\n    assert function() == expected"
        }
    ]
}"""

    def analyze_code(self, code: str, use_fast_mode: bool = False) -> Dict[str, Any]:
        try:
            system_prompt = self.get_test_generation_prompt()
            
            # Use fast configuration if requested
            if use_fast_mode:
                fast_config = self.get_fast_config()
                temp = fast_config['temperature']
                top_p = fast_config['top_p']
                max_tokens = fast_config['max_completion_tokens']
                print(f"🚀 Using FAST mode for Llama-4-Maverick")
            else:
                temp = self.temperature
                top_p = self.top_p
                max_tokens = self.max_completion_tokens
                print(f"🔄 Using STANDARD mode for Llama-4-Maverick")
            
            print(f"📝 Code length: {len(code)} characters")
            print(f"⚙️  Settings: temp={temp}, max_tokens={max_tokens}")

            # Use Groq's chat completion format
            completion = self.client.chat.completions.create(
                model=self.model,
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
                temperature=temp,
                max_completion_tokens=max_tokens,
                top_p=top_p,
                stream=False
            )

            raw_response = completion.choices[0].message.content.strip()
            print(f"✅ Received response: {len(raw_response)} characters")
            
            # Capture debug info for error reporting
            debug_info = {
                'response_length': len(raw_response),
                'response_preview': raw_response[:200] if raw_response else 'No response',
                'response_ending': raw_response[-100:] if len(raw_response) > 100 else raw_response,
                'response_type': str(type(raw_response))
            }
            
            # Use the centralized JSON parser
            from ..utils.json_parser import JSONResponseParser
            result = JSONResponseParser.parse_model_response(raw_response)
            
            if "error" in result:
                # Return comprehensive debug info in error
                return {
                    "error": f"Llama-4-Maverick parsing failed: {result['error']}",
                    "model": "Llama-4-Maverick-17B-128E",
                    "debug_response_preview": debug_info['response_preview'],
                    "debug_response_ending": debug_info['response_ending'],
                    "parser_strategies_tried": result.get('strategies_tried', [])
                }
            
            result["model"] = "Llama-4-Maverick-17B-128E"
            
            return result
                
        except Exception as e:
            return {
                "error": f"Llama-4-Maverick error: {str(e)}",
                "model": "Llama-4-Maverick-17B-128E"
            } 
