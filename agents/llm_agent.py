"""
Base LLM Agent class for the Screenplay Parser App
"""
import logging # Ensure logging is imported
import logging
import re
import json
import time
import requests
import streamlit as st
from typing import Optional, Dict, List, Any, Type, TypeVar, Union

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
                    import unicodedata
                    char_codes = [f"{ord(c)} ({unicodedata.name(c, 'UNKNOWN')})" for c in cleaned_response[:5]] if cleaned_response else []
                    logging.info(f"Cleaned response before list parsing (first 5 chars: {char_codes}): '{cleaned_response[:200]}...' ({len(cleaned_response)} chars)")
                    json_data = json.loads(cleaned_response)
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
                    import unicodedata
                    char_codes = [f"{ord(c)} ({unicodedata.name(c, 'UNKNOWN')})" for c in cleaned_response[:5]] if cleaned_response else []
                    logging.info(f"Cleaned response before object parsing (first 5 chars: {char_codes}): '{cleaned_response[:200]}...' ({len(cleaned_response)} chars)")
                    # Parse and validate a single object
                    json_data = json.loads(cleaned_response)
                    
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
        
        Important: The response MUST be valid JSON with no extra text or markdown.
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
    // IMPORTANT: The response MUST be a valid, complete JSON array
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
                if system_prompt:
                    full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nImportant: Support full UTF-8 encoding for Slovak characters (č, ď, ľ, š, ť, ž, ý, á, í, é, etc.). Respond with JSON only."
                else:
                    full_prompt = f"User: {prompt}\n\nImportant: Support full UTF-8 encoding for Slovak characters (č, ď, ľ, š, ť, ž, ý, á, í, é, etc.). Respond with JSON only."
            
            data = {
                "model": self.model,
                "prompt": full_prompt,
                "temperature": 0.01,       # Very low temperature for structured output
                "num_predict": 12000,       # Equivalent to max_tokens in OpenAI
                "stop": ["```", "```json"], # Stop sequences to avoid markdown wrapping
                "stream": False            # Ensure we get a complete response at once
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
        import openai
        
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

    IMPORTANT: Respond with valid, parseable JSON only. No explanations or other text.
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
    IMPORTANT: Respond with valid, parseable JSON only. No explanations or other text.
    """
                
            data = {
                "model": self.model,
                "prompt": full_prompt,
                "temperature": 0.01,  # Lower temperature for more predictable JSON
                "stream": False       # Ensure we get a complete response at once
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
        """Clean the LLM response to extract useful content."""
        # Handle None response
        if response is None:
            return "[]"  # Return empty JSON array for None
            
        # Remove style tags or any XML/HTML tags
        response = re.sub(r'<[^>]+>', '', response)
        
        # Remove markdown code block syntax
        response = re.sub(r'```json|```python|```|~~~', '', response)
        
        # Remove any "JSON:" or similar prefixes
        response = re.sub(r'^.*?(\[|\{)', r'\1', response, flags=re.DOTALL)
        
        # If we have text after the JSON, remove it
        json_end_match = re.search(r'(\]|\})[^\]\}]*$', response)
        if json_end_match:
            end_pos = json_end_match.start() + 1
            response = response[:end_pos]
        
        # If response starts with backticks, remove them
        response = response.strip('`')

        # Attempt to fix invalid control characters within strings (common issue)
        # Replace unescaped newlines, tabs, etc., with their escaped versions.
        # This is a common cause of JSONDecodeError: Invalid control character
        response = response.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        # Handle cases where the LLM might have already escaped them, causing double escapes
        response = response.replace('\\\\n', '\\n').replace('\\\\r', '\\r').replace('\\\\t', '\\t')
        
        # Fix common JSON errors
        response = response.replace("'", '"')  # Replace single quotes with double quotes
        response = re.sub(r',\s*(\}|\])', r'\1', response)  # Remove trailing commas
        
        return response.strip()