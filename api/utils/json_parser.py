"""
JSON Response Parser for AI Models
Handles common JSON formatting issues from AI model responses
"""

import json
import re
from typing import Dict, Any, Tuple


class JSONResponseParser:
    """Utility class to parse and fix JSON responses from AI models."""
    
    @staticmethod
    def clean_json_string(json_str: str) -> str:
        """Remove markdown code block markers and clean the JSON string."""
        if not json_str:
            return json_str
            
        # Remove markdown code block markers if present
        json_str = json_str.replace("```json", "").replace("```", "")
        
        # Remove leading/trailing whitespace
        json_str = json_str.strip()
        
        # Find JSON object boundaries
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = json_str[start_idx:end_idx + 1]
        
        return json_str
    
    @staticmethod
    def fix_common_json_issues(json_str: str) -> str:
        """Fix common JSON formatting issues."""
        if not json_str:
            return json_str
            
        # Fix control characters FIRST - this is critical for Llama model
        json_str = JSONResponseParser._fix_control_characters(json_str)
        
        # Fix string quote escaping SECOND - critical for Llama model
        json_str = JSONResponseParser._fix_string_escaping(json_str)
        
        # Fix missing commas and delimiters - critical for Llama model
        json_str = JSONResponseParser._fix_missing_commas(json_str)
            
        # Remove trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix unquoted keys (more comprehensive)
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # Fix single quotes in JSON values - improved pattern!
        # This handles cases like: "input": {"x": 'a', "y": 3}
        # We need to be more careful about context
        
        # First, fix single quotes that are clearly JSON string values
        # Pattern: : 'value' (with optional whitespace)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        # Also fix single quotes in object/array contexts
        # Pattern: { "key": 'value' } or [ 'value' ]
        json_str = re.sub(r'(\{\s*"[^"]*"\s*:\s*)\'([^\']*)\'\s*([,}])', r'\1"\2"\3', json_str)
        json_str = re.sub(r'(\[\s*)\'([^\']*)\'\s*([,\]])', r'\1"\2"\3', json_str)
        
        # Handle mixed quotes in object values like {"x": 'a', "y": 3}
        json_str = re.sub(r'(\{\s*"[^"]*"\s*:\s*)\'([^\']*)\'\s*(,\s*"[^"]*"\s*:\s*[^}]*\})', r'\1"\2"\3', json_str)
        
        # Handle problematic escape sequences in string values
        # Fix common problematic patterns like \\n, \\', etc.
        json_str = re.sub(r'\\n', '\\\\n', json_str)  # Fix newline escapes
        json_str = re.sub(r'\\\'', "\\'", json_str)   # Fix single quote escapes
        json_str = re.sub(r'\\"', '\\\\"', json_str)  # Fix double quote escapes
        
        # Fix multiple spaces and newlines
        json_str = re.sub(r'\n\s*\n', '\n', json_str)
        json_str = re.sub(r'  +', ' ', json_str)
        
        # Fix missing quotes around string values (but be careful with complex strings)
        # Only apply to simple alphanumeric values
        json_str = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_\s]*)\s*([,}])', r': "\1"\2', json_str)
        
        # Fix Python None values to JSON null
        json_str = re.sub(r'\bNone\b', 'null', json_str)
        
        # Fix boolean values that might be quoted
        json_str = re.sub(r':\s*"(true|false|null)"\s*', r': \1 ', json_str)
        
        # Fix numbers that might be quoted
        json_str = re.sub(r':\s*"(\d+(?:\.\d+)?)"\s*', r': \1 ', json_str)
        
        return json_str
    
    @staticmethod
    def _fix_control_characters(json_str: str) -> str:
        """Fix invalid control characters in JSON strings - ultra-aggressive approach."""
        if not json_str:
            return json_str
        
        # ULTRA-AGGRESSIVE: Replace ALL control characters with their escaped equivalents
        # This is the nuclear option but should work for any AI model output
        
        result = []
        i = 0
        in_string = False
        
        while i < len(json_str):
            char = json_str[i]
            
            # Track if we're inside a string
            if char == '"' and (i == 0 or json_str[i-1] != '\\'):
                in_string = not in_string
                result.append(char)
            elif in_string:
                # Inside a string - escape control characters
                if char == '\n':
                    result.append('\\n')
                elif char == '\t':
                    result.append('\\t')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\b':
                    result.append('\\b')
                elif char == '\f':
                    result.append('\\f')
                elif char == '\\' and i + 1 < len(json_str) and json_str[i + 1] in 'ntrfb"\'\\':
                    # Already escaped - keep as is
                    result.append(char)
                elif ord(char) < 32:
                    # Skip other control characters entirely
                    pass
                else:
                    result.append(char)
            else:
                # Outside string - only allow safe control characters
                if ord(char) >= 32 or char in '\n\t ':
                    result.append(char)
                # Skip other control characters
            
            i += 1
        
        return ''.join(result)

    @staticmethod
    def _fix_string_escaping(json_str: str) -> str:
        """Fix quote escaping issues within JSON string values."""
        if not json_str:
            return json_str
        
        # Simple approach: find and replace unescaped quotes in JSON string values
        # Look for ": "content with unescaped quotes"
        lines = json_str.split('\n')
        fixed_lines = []
        
        for line in lines:
            # If this line contains a JSON string value with internal quotes
            if '": "' in line and line.count('"') > 2:
                # Find the value part after ": "
                parts = line.split('": "', 1)
                if len(parts) == 2:
                    key_part = parts[0] + '": "'
                    value_part = parts[1]
                    
                    # Find where the string value ends (look for closing quote followed by comma/brace)
                    if re.search(r'"[,\}\]\s]*$', value_part):
                        # Split the value from the ending
                        match = re.search(r'(.*)"([,\}\]\s]*)$', value_part)
                        if match:
                            actual_value = match.group(1)
                            ending = '"' + match.group(2)
                            
                            # Escape any unescaped quotes in the actual value
                            # Be careful not to double-escape already escaped quotes
                            escaped_value = re.sub(r'(?<!\\)"', r'\\"', actual_value)
                            line = key_part + escaped_value + ending
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    @staticmethod
    def _fix_missing_commas(json_str: str) -> str:
        """Fix missing commas and delimiters in JSON - common AI model issue."""
        if not json_str:
            return json_str
        
        lines = json_str.split('\n')
        result_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Current line for adding to result
            current_line = line
            
            # Look ahead to next non-empty line
            next_line_content = None
            for j in range(i + 1, len(lines)):
                next_stripped = lines[j].strip()
                if next_stripped:
                    next_line_content = next_stripped
                    break
            
            if next_line_content:
                # Case 1: String value followed by object start - add comma after quote
                if (stripped.endswith('"') and 
                    next_line_content.startswith('{') and 
                    not stripped.endswith(',"') and
                    not stripped.endswith(': "') and
                    ':' in stripped):
                    current_line = current_line.rstrip() + ','
                
                # Case 2: Object end followed by object start - add comma after }
                elif (stripped.endswith('}') and 
                      next_line_content.startswith('{') and
                      not stripped.endswith(',}')):
                    current_line = current_line.rstrip() + ','
                
                # Case 3: String value followed by property - add comma after quote
                elif (stripped.endswith('"') and 
                      ':' in next_line_content and
                      next_line_content.startswith('"') and
                      not stripped.endswith(',"') and
                      ':' in stripped):
                    current_line = current_line.rstrip() + ','
                    
                # Case 4: Object end followed by property - add comma after }
                elif (stripped.endswith('}') and 
                      ':' in next_line_content and
                      next_line_content.startswith('"') and
                      not stripped.endswith(',}')):
                    current_line = current_line.rstrip() + ','
                
                # Case 5: Array end followed by property - add comma after ]
                elif (stripped.endswith(']') and 
                      ':' in next_line_content and
                      next_line_content.startswith('"') and
                      not stripped.endswith(',]')):
                    current_line = current_line.rstrip() + ','
                    
                # Case 6: Number/boolean followed by property - add comma
                elif (re.match(r'.*[0-9]+\s*$|.*true\s*$|.*false\s*$|.*null\s*$', stripped) and
                      ':' in next_line_content and
                      next_line_content.startswith('"')):
                    current_line = current_line.rstrip() + ','
            
            result_lines.append(current_line)
        
        return '\n'.join(result_lines)

    @staticmethod
    def extract_json_from_text(text: str) -> str:
        """Extract JSON object from mixed text content."""
        if not text:
            return "{}"
            
        # Try multiple patterns to find JSON
        patterns = [
            # Standard JSON object with nested objects/arrays
            r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}',
            # JSON with arrays
            r'\{[^{}]*\[[^\[\]]*(?:\{[^{}]*\}[^\[\]]*)*\][^{}]*\}',
            # Simple object pattern
            r'\{[^{}]+\}',
            # Any content between curly braces
            r'\{.*?\}'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # Return the largest match (most complete JSON)
                largest_match = max(matches, key=len)
                if len(largest_match) > 10:  # Must be substantial
                    return largest_match
        
        # Try to find JSON-like content manually
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    return text[start_idx:i+1]
        
        # If no JSON found, try to create a basic structure
        return "{}"
    
    @staticmethod
    def validate_json(json_str: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        Validate JSON string and return parsing result.
        Returns: (is_valid, parsed_data, error_message)
        """
        try:
            parsed_data = json.loads(json_str)
            return True, parsed_data, ""
        except json.JSONDecodeError as e:
            return False, {}, str(e)
    
    @staticmethod
    def parse_model_response(raw_response: str) -> Dict[str, Any]:
        """
        Parse AI model response with comprehensive error handling.
        Returns a dictionary with parsed data or error information.
        """
        if not raw_response or not raw_response.strip():
            return {
                "error": "Empty response from model",
                "raw_output": raw_response
            }
        
        # Try multiple parsing strategies
    @staticmethod
    def parse_model_response(raw_response: str) -> Dict[str, Any]:
        """
        Parse AI model response with comprehensive error handling.
        Returns a dictionary with parsed data or error information.
        """
        if not raw_response or not raw_response.strip():
            return {
                "error": "Empty response from model",
                "raw_output": raw_response
            }
        
    @staticmethod
    def parse_model_response(raw_response: str) -> Dict[str, Any]:
        """
        Parse AI model response with comprehensive error handling.
        Returns a dictionary with parsed data or error information.
        """
        if not raw_response or not raw_response.strip():
            return {
                "error": "Empty response from model",
                "raw_output": raw_response
            }
        
    @staticmethod
    def parse_model_response(raw_response: str) -> Dict[str, Any]:
        """
        Parse AI model response with comprehensive error handling.
        Returns a dictionary with parsed data or error information.
        """
        if not raw_response or not raw_response.strip():
            return {
                "error": "Empty response from model",
                "raw_output": raw_response
            }
        
        # Temporary debug for GPT-4 issues - DISABLED
        debug_mode = False
        if debug_mode and len(raw_response) > 2000:  # Likely a substantial response
            print(f"\n🔍 PARSING DEBUG: Response length: {len(raw_response)}")
            print(f"First 200 chars: {repr(raw_response[:200])}")
            print(f"Last 200 chars: {repr(raw_response[-200:])}")
        
        # Try multiple parsing strategies
        strategies = [
            ("direct_parse", lambda x: JSONResponseParser._try_direct_parse(x)),
            ("clean_and_parse", lambda x: JSONResponseParser._try_clean_and_parse(x)),
            ("python_eval_parse", lambda x: JSONResponseParser._try_python_eval_parse(x)),  # New Python expression strategy
            ("extract_and_parse", lambda x: JSONResponseParser._try_extract_and_parse(x)),
            ("llama_special_parse", lambda x: JSONResponseParser._try_llama_special_parse(x)),  # New Llama-specific strategy
            ("alternative_parse", lambda x: JSONResponseParser._try_alternative_parsing(x))
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                result = strategy_func(raw_response)
                if result and "error" not in result:
                    return result
            except Exception:
                continue
        
        # If all strategies fail, return comprehensive debug info
        return {
            "error": f"All parsing strategies failed",
            "raw_output": raw_response,
            "strategies_tried": [s[0] for s in strategies]
        }
    
    @staticmethod
    def _try_python_eval_parse(raw_response: str) -> Dict[str, Any]:
        """Try parsing by safely evaluating Python expressions in JSON."""
        try:
            # Step 1: Clean the response
            cleaned = JSONResponseParser.clean_json_string(raw_response)
            fixed = JSONResponseParser.fix_common_json_issues(cleaned)
            
            # Step 2: Replace Python expressions with evaluated results
            def replace_python_expressions(match):
                expr = match.group(0)
                try:
                    # Only evaluate safe expressions
                    if any(safe_pattern in expr for safe_pattern in ['range(', 'for i in', 'for x in']):
                        # Evaluate the expression safely
                        result = eval(expr)
                        # Convert to JSON-compatible format
                        if isinstance(result, list):
                            return json.dumps(result)
                        else:
                            return str(result)
                except:
                    pass
                return expr
            
            # Replace various Python list comprehension patterns
            # Pattern 1: [i for i in range(...)]
            fixed = re.sub(r'\[i for i in range\([^]]*\)\]', replace_python_expressions, fixed)
            
            # Pattern 2: [i * 10 for i in range(...)]
            fixed = re.sub(r'\[i \* \d+ for i in range\([^]]*\)\]', replace_python_expressions, fixed)
            
            # Pattern 3: [expression for variable in range(...)]
            fixed = re.sub(r'\[[^]]*for [a-zA-Z_]\w* in range\([^]]*\)\]', replace_python_expressions, fixed)
            
            # Pattern 4: list(range(...)) expressions
            fixed = re.sub(r'list\(range\([^)]*\)\)', replace_python_expressions, fixed)
            
            # Step 3: Try parsing the result
            parsed_data = json.loads(fixed)
            return JSONResponseParser._validate_and_format(parsed_data, raw_response)
            
        except Exception as e:
            return {"error": f"Python eval parse failed: {str(e)}"}
    
    @staticmethod
    def _fix_escape_sequences_simple(json_str: str) -> str:
        """Simple but effective escape sequence fixing."""
        # For JSON strings that contain Python code, we need to ensure newlines are properly escaped
        
        # Find all string values and fix escape sequences in them
        def fix_string_value(match):
            quote_char = match.group(1)  # Either single or double quote
            content = match.group(2)
            
            # Fix common escape issues
            if '\\n' in content and '\\\\n' not in content:
                content = content.replace('\\n', '\\\\n')
            if '\\t' in content and '\\\\t' not in content:
                content = content.replace('\\t', '\\\\t')
            if '\\r' in content and '\\\\r' not in content:
                content = content.replace('\\r', '\\\\r')
            
            return f'{quote_char}{content}{quote_char}'
        
        # Apply to all quoted strings
        json_str = re.sub(r'(["\'])([^"\']*(?:\\.[^"\']*)*)\1', fix_string_value, json_str)
        
        return json_str
    
    @staticmethod
    def _try_direct_parse(raw_response: str) -> Dict[str, Any]:
        """Try parsing the response directly as JSON."""
        try:
            parsed_data = json.loads(raw_response)
            return JSONResponseParser._validate_and_format(parsed_data, raw_response)
        except json.JSONDecodeError:
            # Try with escape sequence fixes
            try:
                fixed_response = JSONResponseParser._fix_escape_sequences_simple(raw_response)
                parsed_data = json.loads(fixed_response)
                return JSONResponseParser._validate_and_format(parsed_data, raw_response)
            except json.JSONDecodeError:
                return {"error": "Direct JSON parsing failed"}
    
    @staticmethod
    def _try_clean_and_parse(raw_response: str) -> Dict[str, Any]:
        """Clean the response and try parsing."""
        try:
            # Step 1: Clean the response
            cleaned = JSONResponseParser.clean_json_string(raw_response)
            
            # Step 2: Fix common JSON issues including control characters
            fixed = JSONResponseParser.fix_common_json_issues(cleaned)
            
            # Step 3: Try parsing the fixed version first
            try:
                parsed_data = json.loads(fixed)
                return JSONResponseParser._validate_and_format(parsed_data, raw_response, cleaned, fixed)
            except json.JSONDecodeError:
                pass
            
            # Step 4: Try fixing Python tuples (Phi model specific issue)
            try:
                tuple_fixed = JSONResponseParser._fix_python_tuples(cleaned)
                parsed_data = json.loads(tuple_fixed)
                return JSONResponseParser._validate_and_format(parsed_data, raw_response, cleaned, tuple_fixed)
            except json.JSONDecodeError:
                pass
            
            # Step 5: Try parsing cleaned version without fixes
            try:
                parsed_data = json.loads(cleaned)
                return JSONResponseParser._validate_and_format(parsed_data, raw_response, cleaned)
            except json.JSONDecodeError:
                pass
            
            # Step 6: Try with escape sequence fixes on the fixed version
            try:
                escape_fixed = JSONResponseParser._fix_escape_sequences_simple(fixed)
                parsed_data = json.loads(escape_fixed)
                return JSONResponseParser._validate_and_format(parsed_data, raw_response, cleaned, escape_fixed)
            except json.JSONDecodeError:
                pass
            
            # Step 7: Final attempt - should not fail if control characters were the issue
            parsed_data = json.loads(fixed)
            return JSONResponseParser._validate_and_format(parsed_data, raw_response, cleaned, fixed)
        except json.JSONDecodeError as e:
            return {"error": f"Clean and parse failed: {e}"}
    
    @staticmethod
    def _try_extract_and_parse(raw_response: str) -> Dict[str, Any]:
        """Extract JSON from text and parse."""
        try:
            # Extract JSON if embedded in text
            extracted = JSONResponseParser.extract_json_from_text(raw_response)
            
            # Fix common issues
            fixed = JSONResponseParser.fix_common_json_issues(extracted)
            
            # Validate and parse
            parsed_data = json.loads(fixed)
            return JSONResponseParser._validate_and_format(parsed_data, raw_response, extracted, fixed)
        except json.JSONDecodeError as e:
            return {"error": f"Extract and parse failed: {e}"}
    
    @staticmethod
    def _validate_and_format(parsed_data: Dict[str, Any], raw_response: str, 
                           cleaned: str = "", fixed: str = "") -> Dict[str, Any]:
        """Validate parsed data and format the response."""
        if not isinstance(parsed_data, dict):
            return {"error": "Response is not a JSON object"}
        
        # Ensure required fields exist
        required_fields = ["test_cases", "test_framework", "coverage_areas"]
        missing_fields = [field for field in required_fields if field not in parsed_data]
        
        if missing_fields:
            # Try to create default structure
            if "test_cases" not in parsed_data:
                parsed_data["test_cases"] = []
            if "test_framework" not in parsed_data:
                parsed_data["test_framework"] = "pytest"
            if "coverage_areas" not in parsed_data:
                parsed_data["coverage_areas"] = ["basic functionality"]
        
        # Return the properly parsed JSON object
        return parsed_data

    @staticmethod
    def _try_llama_special_parse(raw_response: str) -> Dict[str, Any]:
        """Special parsing strategy for Llama model with step-by-step fixes."""
        try:
            # Step 1: Extract JSON from text
            extracted = JSONResponseParser.extract_json_from_text(raw_response)
            
            # Step 2: Clean the response  
            cleaned = JSONResponseParser.clean_json_string(extracted)
            
            # Step 3: Apply fixes in the exact order that works for Llama
            # Fix control characters first
            control_fixed = JSONResponseParser._fix_control_characters(cleaned)
            
            # Fix string escaping second 
            escaping_fixed = JSONResponseParser._fix_string_escaping(control_fixed)
            
            # Fix missing commas third
            comma_fixed = JSONResponseParser._fix_missing_commas(escaping_fixed)
            
            # Step 4: Try parsing the fully fixed version
            parsed_data = json.loads(comma_fixed)
            return JSONResponseParser._validate_and_format(parsed_data, raw_response, cleaned, comma_fixed)
            
        except json.JSONDecodeError as e:
            return {"error": f"Llama special parse failed: {e}"}
        except Exception as e:
            return {"error": f"Llama special parse exception: {e}"}
    
    @staticmethod
    def _try_alternative_parsing(raw_response: str) -> Dict[str, Any]:
        """Try alternative parsing methods for non-standard responses."""
        try:
            # Method 1: Look for test case patterns in text
            test_cases = []
            lines = raw_response.split('\n')
            
            current_test = {}
            for line in lines:
                line = line.strip()
                if line.startswith('Test') and ':' in line:
                    if current_test:
                        test_cases.append(current_test)
                    current_test = {"name": line.split(':', 1)[1].strip()}
                elif line.startswith('Description:'):
                    current_test["description"] = line.split(':', 1)[1].strip()
                elif line.startswith('Input:'):
                    current_test["input"] = line.split(':', 1)[1].strip()
                elif line.startswith('Expected:'):
                    current_test["expected_output"] = line.split(':', 1)[1].strip()
                elif line.startswith('Code:'):
                    current_test["test_code"] = line.split(':', 1)[1].strip()
            
            if current_test:
                test_cases.append(current_test)
            
            if test_cases:
                result = {
                    "test_cases": test_cases,
                    "test_framework": "pytest",
                    "coverage_areas": ["extracted from text"]
                }
                return {
                    "test_cases": json.dumps(result, indent=2),
                    "raw_output": raw_response,
                    "parsing_method": "alternative_text_extraction"
                }
        
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def _fix_python_tuples(json_str: str) -> str:
        """Fix Python tuples in JSON string by converting (a, b) to [a, b]."""
        import re
        
        # More careful pattern to avoid function calls
        # Only replace tuples that are JSON values (after : or in arrays)
        def replace_tuple(match):
            full_match = match.group(0)
            prefix = match.group(1)
            content = match.group(2)
            
            # Only replace if it looks like a JSON value context
            # Check if preceded by : (JSON value) or , (array element) or [ (array start)
            if prefix and (prefix.endswith(':') or prefix.endswith(',') or prefix.endswith('[')):
                # Only replace if content looks like data (has commas and quotes/numbers)
                if ',' in content and ('"' in content or "'" in content or any(c.isdigit() for c in content)):
                    return f'{prefix} [{content}]'
            
            return full_match  # Keep original
        
        # Pattern to match context before parentheses
        pattern = r'([:\s,\[])\s*\(([^)]+)\)'
        
        return re.sub(pattern, replace_tuple, json_str)