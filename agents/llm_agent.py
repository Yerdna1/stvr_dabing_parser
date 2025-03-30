"""
Base LLM Agent class for the Screenplay Parser App
"""
import logging # Ensure logging is imported
import re
import json
import time
import requests
import streamlit as st
import openai
import unicodedata
from typing import Optional, Dict, List, Any, Type, TypeVar, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import MAX_RETRIES, RETRY_DELAY, TEMPERATURE
from models import BaseSegment, ProcessedSegment, Entities


T = TypeVar('T')

class LLMAgent:
    """Base class for LLM agents"""
    
    def __init__(self, provider: str, model: str, api_key: Optional[str] = None, ollama_url: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.ollama_url = ollama_url
        
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call the LLM with the given prompt and return the response."""
        for attempt in range(MAX_RETRIES):
            try:
                if self.provider == "OpenAI":
                    return self._call_openai(prompt, system_prompt)
                elif self.provider == "DeepSeek" or (self.provider == "Ollama" and "deepseek" in self.model.lower()):
                    # Use special DeepSeek handling
                    return self._call_deepseek(prompt, system_prompt)
                elif self.provider == "Ollama":
                    return self._call_ollama(prompt, system_prompt)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    if hasattr(st, 'session_state') and st.session_state.get('detailed_progress', True):
                        st.warning(f"Retrying after error: {str(e)}")
                    continue
                else:
                    st.error(f"Error calling LLM: {str(e)}")
                    return ""

    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type(json.JSONDecodeError))
    def _parse_json_with_retry(self, json_str: str) -> Any:
        """Parse JSON with retries and error handling."""
        try:
            # Try standard JSON parse first
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # If that fails, try to fix common issues and retry
            logging.warning(f"Initial JSON parsing failed: {str(e)}. Attempting to fix JSON.")
            fixed_json = self._fix_json_string(json_str)
            logging.info(f"Repaired JSON (first 100 chars): {fixed_json[:100]}")
            return json.loads(fixed_json)

    def _fix_json_string(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        if not json_str:
            return "[]"
            
        # Replace literal \n and \t with actual newlines and tabs
        json_str = json_str.replace('\\n', '\n').replace('\\t', '\t')
        
        # Remove extra escaping within strings
        json_str = json_str.replace('\\"', '"')
        
        # Add braces if needed 
        if not json_str.strip().startswith('[') and not json_str.strip().startswith('{'):
            json_str = '[' + json_str
        if not json_str.strip().endswith(']') and not json_str.strip().endswith('}'):
            json_str = json_str + ']'
            
        # Fix trailing commas
        json_str = re.sub(r',\s*(\}|\])', r'\1', json_str)
        
        # Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')
        
        return json_str

    def _call_llm_with_schema(self, prompt: str, model_type, system_prompt: Optional[str] = None, is_list: bool = False):
        """Call LLM and validate response against a Pydantic schema"""
        try:
            # Enhance prompt with schema information
            prompt_with_schema = self._enhance_prompt_with_schema(prompt, model_type, is_list)
            
            # Add system prompt for better guidance
            enhanced_system_prompt = system_prompt
            if system_prompt:
                enhanced_system_prompt += "\n\nYou must respond with valid JSON matching the required schema."
            else:
                enhanced_system_prompt = "You must respond with valid JSON matching the required schema."
            
            # Always add this instruction for better JSON responses
            enhanced_system_prompt += "\n\nIMPORTANT: Return a valid JSON array with no extra text. Do not use escaped newlines or tabs in your JSON."
            
            # Call the LLM with the enhanced prompt
            response = self._call_llm(prompt_with_schema, enhanced_system_prompt)
            
            # Check for None responses
            if response is None:
                st.error("Received None response from LLM")
                logging.error("Received None response from LLM")
                return [] if is_list else {}
                
            # Clean the response
            cleaned_response = self._clean_response(response)
            
            # Parse JSON
            if is_list:
                try:
                    # Log first few character codes for debugging
                    char_codes = [f"{ord(c)} ({unicodedata.name(c, 'UNKNOWN')})" for c in cleaned_response[:5]] if cleaned_response else []
                    logging.info(f"Cleaned response before list parsing (first 5 chars: {char_codes}): '{cleaned_response[:200]}...' ({len(cleaned_response)} chars)")
                    
                    # Use the retry method for more robust parsing
                    json_data = self._parse_json_with_retry(cleaned_response)
                    
                    if not isinstance(json_data, list):
                        json_data = [json_data]
                        
                    # Otherwise use standard Pydantic validation
                    validated_items = []
                    for item in json_data:
                        # Check for None or empty values in crucial fields
                        for key in ['text', 'speaker', 'timecode']:
                            if key in item and item[key] is None:
                                item[key] = ""  # Convert None to empty string
                        
                        try:
                            validated_item = model_type.model_validate(item)
                            validated_items.append(validated_item.model_dump())
                        except Exception as e:
                            st.warning(f"Item validation error for {model_type.__name__}: {str(e)}")
                            logging.warning(f"Item validation error for {model_type.__name__}: {str(e)}")
                            # If validation fails, just add the original item
                            validated_items.append(item)
                    
                    return validated_items
                except json.JSONDecodeError as e:
                    st.error(f"JSON parsing error: {str(e)}")
                    logging.error(f"JSON parsing error: {str(e)}")
                    return []
            else:
                try:
                    # Log first few character codes for debugging
                    char_codes = [f"{ord(c)} ({unicodedata.name(c, 'UNKNOWN')})" for c in cleaned_response[:5]] if cleaned_response else []
                    logging.info(f"Cleaned response before object parsing (first 5 chars: {char_codes}): '{cleaned_response[:200]}...' ({len(cleaned_response)} chars)")
                    
                    # Parse and validate a single object using retry method
                    json_data = self._parse_json_with_retry(cleaned_response)

                    # Handle case where parsing returns a list with a single object inside
                    if isinstance(json_data, list) and len(json_data) == 1 and isinstance(json_data[0], dict):
                        logging.info("Extracted single dictionary from list for object validation.")
                        json_data = json_data[0]
                    elif not isinstance(json_data, dict):
                        st.error(f"Expected a dictionary for single object validation, but got {type(json_data)}")
                        logging.error(f"Expected a dictionary for single object validation, but got {type(json_data)}")
                        return {}

                    # Check for None values
                    for key in ['text', 'speaker', 'timecode']:
                        if key in json_data and json_data[key] is None:
                            json_data[key] = ""  # Convert None to empty string
                    
                    validated_data = model_type.model_validate(json_data)
                    return validated_data.model_dump()
                except json.JSONDecodeError:
                    st.error(f"JSON parsing error in single object")
                    logging.error(f"JSON parsing error in single object")
                    return {}
                    
        except Exception as e:
            st.error(f"Validation error for {model_type.__name__ if model_type else 'unknown model'}: {str(e)}")
            logging.error(f"Validation error for {model_type.__name__ if model_type else 'unknown model'}: {str(e)}")
            # Return the original cleaned response as a fallback
            return [] if is_list else {}
    
    def _enhance_prompt_with_schema(self, prompt: str, model_type, is_list: bool) -> str:
        """Add schema information to the prompt for better LLM parsing"""
        try:
            # Try Pydantic v2.x method first
            schema_json = model_type.model_json_schema()
        except AttributeError:
            # Fall back to Pydantic v1.x method
            schema_json = model_type.schema_json(indent=2)
            
        if isinstance(schema_json, str):
            schema_str = schema_json
        else:
            schema_str = json.dumps(schema_json, indent=2)
        
        enhanced_prompt = f"""
        {prompt}
        
        Please format your response as valid JSON that matches this schema:
        {schema_str}
        
        {"The response should be a JSON array of objects matching this schema." if is_list else "The response should be a single JSON object matching this schema."}
        
        IMPORTANT: 
        - The response MUST be valid JSON with no extra text or markdown
        - Do not use escaped newlines or tabs within the JSON
        - Return a properly formatted JSON array only, with no explanation text before or after
        """
        return enhanced_prompt

    def _call_deepseek(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Ollama API with optimized handling for DeepSeek Coder models."""
        try:
            url = f"{self.ollama_url}/api/generate"
            
            # DeepSeek Coder prompt optimization - function format works better
            code_prompt = f"""// Function to parse screenplay text
    function parseScreenplaySegment(text) {{
    // Input text to parse:
    /*
    {prompt.replace('```', '').strip()}
    */

    // Return a JSON array with parsed elements (scene headers, dialogue, etc.)
    // IMPORTANT: The response MUST be ONLY a valid, complete JSON array.
    // IMPORTANT: DO NOT include any explanations, comments, code, or markdown.
    // IMPORTANT: Support full UTF-8 encoding for Slovak characters (č, ď, ľ, š, ť, ž, ý, á, í, é, etc.)
    return [
        // Example format - to be replaced with actual parsed content:
        {{
        "type": "scene_header",
        "scene_type": "INT",
        "timecode": "00:01:23"
        }},
        {{
        "type": "dialogue",
        "character": "CHARACTER_NAME",
        "audio_type": "VO",
        "text": "Dialogue text with Slovak characters: čšťžýáíéľďň"
        }}
    ];
    }}

    // Call the function and return ONLY the JSON result:
    parseScreenplaySegment();
    """

            # Use code-optimized format if selected, otherwise use standard format
            if hasattr(self, 'use_code_format') and self.use_code_format:
                full_prompt = code_prompt
            else:
                system_message = system_prompt if system_prompt else ""
                system_message += "\n\nCRITICAL INSTRUCTION: Your response MUST be ONLY a valid JSON array matching the requested schema. Do NOT include any introductory text, explanations, comments, code snippets, markdown formatting, or anything other than the JSON itself. Ensure full UTF-8 support for all characters, including Slovak (č, ď, ľ, š, ť, ž, ý, á, í, é, etc.)."
                
                full_prompt = f"System: {system_message}\n\nUser: {prompt}"

            data = {
                "model": self.model,
                "prompt": full_prompt,
                "temperature": 0.01,       # Very low temperature for structured output
                "num_predict": 12000,       # Equivalent to max_tokens in OpenAI
                "stop": ["```", "```json"], # Stop sequences to avoid markdown wrapping
                "stream": False,           # Ensure we get a complete response at once
                "format": "json"           # Explicitly request JSON format
            }

            if hasattr(st, 'session_state') and st.session_state.get('detailed_progress', True):
                st.write(f"Calling DeepSeek model: {self.model}")
            
            if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
                st.write(f"DeepSeek prompt length: {len(full_prompt)} characters")
            
            # Make the API call with configured timeout
            timeout_seconds = st.session_state.get('timeout_seconds', 120)
            response = requests.post(url, json=data, timeout=timeout_seconds)
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "")
                
                # Attempt to complete the JSON if it appears to be truncated
                if raw_response.count("[") > raw_response.count("]") or raw_response.count("{") > raw_response.count("}"):
                    raw_response = self._complete_json(raw_response)
                    
                # Aggressive JSON extraction specifically for DeepSeek
                # First try to find array pattern
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', raw_response, re.DOTALL)
                if json_match:
                    raw_response = json_match.group(0)
                else:
                    # If no array found, try to find any valid JSON object
                    obj_match = re.search(r'\{\s*"[^"]+"\s*:.*?\}', raw_response, re.DOTALL)
                    if obj_match:
                        raw_response = f"[{obj_match.group(0)}]"
                
                # Clean the response
                cleaned_response = self._clean_response(raw_response)
                
                # Log the raw and cleaned responses if debug mode is enabled
                if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
                    st.write("Raw DeepSeek response:")
                    st.text(raw_response[:500] + ("..." if len(raw_response) > 500 else ""))
                    st.write("Cleaned response:")
                    st.text(cleaned_response[:500] + ("..." if len(cleaned_response) > 500 else ""))
                    
                return cleaned_response
            else:
                raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to Ollama at {self.ollama_url}. Make sure Ollama is running.")
            return "[]"  # Return empty JSON array on connection error
        except requests.exceptions.Timeout:
            st.warning(f"DeepSeek request timed out after {st.session_state.get('timeout_seconds', 120)} seconds.")
            return "[]"  # Return empty JSON array on timeout
        except Exception as e:
            st.error(f"Error calling DeepSeek: {str(e)}")
            return "[]"  # Return empty JSON array on any error

    def _complete_json(self, partial_json: str) -> str:
        """Attempt to complete truncated JSON responses."""
        # Count brackets to detect incomplete JSON
        open_brackets = partial_json.count("[")
        close_brackets = partial_json.count("]")
        open_braces = partial_json.count("{")
        close_braces = partial_json.count("}")
        
        # First, ensure we have a starting bracket
        if "[" not in partial_json and open_braces > 0:
            partial_json = "[" + partial_json
            open_brackets += 1
        
        # Add missing closing braces
        if open_braces > close_braces:
            partial_json += "}" * (open_braces - close_braces)
        
        # Add missing closing brackets
        if open_brackets > close_brackets:
            partial_json += "]" * (open_brackets - close_brackets)
        
        return partial_json
    
    def _call_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call OpenAI API."""
        openai.api_key = self.api_key
        
        messages = []
        if system_prompt:
            system_prompt += "\n\nImportant: Always preserve and support full UTF-8 encoding for Slovak characters (č, ď, ľ, š, ť, ž, ý, á, í, é, etc.)."
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": "Support full UTF-8 encoding for Slovak characters (č, ď, ľ, š, ť, ž, ý, á, í, é, etc.)."})
        
        messages.append({"role": "user", "content": prompt})
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE
        )
        
        return response.choices[0].message.content
    
    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Ollama API with improved response handling."""
        try:
            url = f"{self.ollama_url}/api/generate"
            
            # Format the prompt properly
            if system_prompt:
                # Use a clear separator between system prompt and user prompt
                full_prompt = f"""SYSTEM INSTRUCTION:
    {system_prompt}
    
    Important: Support full UTF-8 encoding for Slovak characters (č, ď, ľ, š, ť, ž, ý, á, í, é, etc.).

    USER:
    {prompt}

    IMPORTANT: Respond with valid, parseable JSON only. DO NOT include any explanations, comments, code, markdown, or any text outside the JSON structure itself.
    Example format:
    [
    {{
        "type": "scene_header",
        "scene_type": "INT",
        "timecode": "00:01:23"
    }},
    {{
        "type": "dialogue",
        "character": "CHARACTER_NAME",
        "audio_type": "VO",
        "text": "Dialogue text with Slovak characters: čšťžýáíéľďň"
    }}
    ]
    """
            else:
                full_prompt = f"""USER:
    {prompt}

    IMPORTANT: Support full UTF-8 encoding for Slovak characters (č, ď, ľ, š, ť, ž, ý, á, í, é, etc.).
    CRITICAL INSTRUCTION: Respond ONLY with the valid JSON structure. No introductory text, no explanations, no code, no markdown.
    """

            data = {
                "model": self.model,
                "prompt": full_prompt,
                "temperature": 0.01,  # Lower temperature for more predictable JSON
                "stream": False,      # Ensure we get a complete response at once
                "format": "json"      # Explicitly request JSON format
            }

            st.write(f"Calling Ollama model: {self.model}")
            if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
                st.write(f"Prompt length: {len(full_prompt)} characters")
               # st.write(full_prompt)
            
            # Make the API call with extended timeout
            response = requests.post(url, json=data, timeout=180)
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "")
                
                # Try to extract JSON array more aggressively
                json_match = re.search(r'\[\s*\{.*\}\s*\]', raw_response, re.DOTALL)
                if json_match:
                    raw_response = json_match.group(0)
                
                # Clean the response
                cleaned_response = self._clean_response(raw_response)
                
                # Log the raw and cleaned responses if debug mode is enabled
                if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
                    st.write("Raw Ollama response:")
                    st.text(raw_response[:1000] + ("..." if len(raw_response) > 1000 else ""))
                    st.write("Cleaned response:")
                    st.text(cleaned_response[:1000] + ("..." if len(cleaned_response) > 1000 else ""))
                    
                return cleaned_response
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to Ollama at {self.ollama_url}. Make sure Ollama is running.")
            return ""
        except Exception as e:
            st.error(f"Error calling Ollama: {str(e)}")
            return ""

    def _clean_response(self, response: str) -> str:
        """Clean the LLM response to extract the first valid JSON object or array."""
        if response is None:
            return "[]"

        response = response.strip()
        # Remove markdown code blocks aggressively at the beginning and end
        response = re.sub(r'^(```json|```python|```|~~~\s*)', '', response)
        response = re.sub(r'(\s*```|~~~)$', '', response).strip()

        # Defensive Check: Ensure the response starts like JSON
        if not response.startswith('[') and not response.startswith('{'):
            logging.warning(f"LLM response did not start with '[' or '{{'. Assuming invalid format. Response start: {response[:200]}...")
            return "[]" # Return empty list immediately

        # Find the start of the first JSON structure ([ or {)
        first_bracket = response.find('[')
        first_brace = response.find('{')
        start_index = -1

        if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
            start_index = first_bracket
            start_char = '['
            end_char = ']'
        elif first_brace != -1:
            start_index = first_brace
            start_char = '{'
            end_char = '}'
        else:
            logging.warning(f"No JSON object or array start found in raw response: {response[:200]}...")
            return "[]"

        # Try to parse incrementally to find the end of the first valid JSON structure
        # Start checking from minimum possible length (e.g., '{}' or '[]')
        min_len = 2
        valid_json_part = ""

        for i in range(start_index + min_len, len(response) + 1):
            substring = response[start_index:i]
            # Optimization: Only attempt parsing if the substring ends with the potential closing character
            if substring.endswith(end_char):
                try:
                    # Attempt to parse this potentially complete substring
                    json.loads(substring)
                    # If successful, this is the first valid complete JSON structure
                    valid_json_part = substring
                    logging.info(f"Successfully extracted valid JSON prefix of length {len(valid_json_part)}")
                    break # IMPORTANT: Stop searching immediately
                except json.JSONDecodeError:
                    # It looked complete but wasn't valid JSON (e.g., nested structure incomplete), keep extending
                    continue

        if not valid_json_part:
            logging.warning(f"Could not extract a valid JSON prefix via incremental parsing. Response start: {response[start_index:start_index+200]}... Falling back to _fix_json_string on the whole response.")
            # Fallback: Try running the basic fixer on the whole string from the start index
            # This is a last resort before _parse_json_with_retry potentially fails
            valid_json_part = self._fix_json_string(response[start_index:])
            # We return this potentially fixed string, and let _parse_json_with_retry handle the final parse attempt/error
            logging.warning(f"Returning fallback fixed string: {valid_json_part[:200]}...")
            # Note: We don't apply further cleaning here as _fix_json_string already did basic steps.
            return valid_json_part


        # --- Apply final cleaning steps ONLY to the successfully extracted valid_json_part ---
        # These steps refine the structurally valid JSON but shouldn't break it

        # Normalize internal whitespace carefully - avoid breaking strings
        # This is less critical now that we have a valid structure, focus on other fixes
        # valid_json_part = re.sub(r'\s+', ' ', valid_json_part).strip() # Maybe too aggressive?

        valid_json_part = valid_json_part.replace("'", '"') # Replace single quotes
        valid_json_part = re.sub(r',\s*(\}|\])', r'\1', valid_json_part) # Remove trailing commas
        valid_json_part = re.sub(r'\}\s*\{', '}, {', valid_json_part) # Add missing commas between objects

        if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
            logging.debug(f"Original response length: {len(response)}")
            logging.debug(f"Extracted JSON part length: {len(valid_json_part)}")
            logging.debug(f"Cleaned JSON part (first 200): {valid_json_part[:200]}")

        return valid_json_part
