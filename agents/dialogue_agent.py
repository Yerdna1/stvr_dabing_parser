"""
Dialogue Processing Agent for screenplay parsing
"""
import re
import json
import streamlit as st
from typing import Dict, List, Any

from agents.llm_agent import LLMAgent
from models import ProcessedSegment, DialogueSegment

class DialogueProcessingAgent(LLMAgent):
    """Agent for understanding and normalizing dialogue."""
    
    def process_dialogue(self, dialogue_segments: List[Dict]) -> List[Dict]:
        """Process dialogue segments to normalize and fill in missing information using Pydantic-AI."""
        if not dialogue_segments:
            return []
            
        system_prompt = """
        You are an expert dialogue analyzer for screenplays. Your task is to process dialogue segments,
        ensuring consistency in character names and audio notations.
        
        For each dialogue segment:
        1. Normalize character names to their canonical form
        2. Identify the correct audio notation type
        3. Clean the dialogue text, preserving stage directions within dialogue
        
        Return the processed dialogues as a JSON array of objects with these fields:
        - type: "dialogue" for all segments
        - speaker: The normalized speaker name with any audio notation in parentheses
        - text: The cleaned dialogue text
        - timecode: Any timestamp if present
        """
        
        # Process in batches to avoid hitting token limits
        batch_size = 20
        all_processed = []
        
        for i in range(0, len(dialogue_segments), batch_size):
            batch = dialogue_segments[i:i+batch_size]
            st.write(f"Processing dialogue batch {i//batch_size + 1}/{(len(dialogue_segments)-1)//batch_size + 1}...")
            
            prompt = f"""
            Process these dialogue segments from a screenplay to normalize character names,
            audio notations, and clean up dialogue text.
            
            ```
            {json.dumps(batch, indent=2)}
            ```
            
            For each segment:
            1. Standardize the speaker name (e.g., "JOHN" instead of "JOHN1" or "john")
            2. Keep audio notations like (VO) or (MO) with the speaker name
            3. Clean any formatting issues in the dialogue text
            4. Preserve any timecode information
            
            Return a list of processed dialogue segments.
            """
            
            # First try: Use Pydantic-AI for validation
            try:
                # Use the call_llm_with_schema method to get validated dialogue segments
                processed_batch = self._call_llm_with_schema(prompt, ProcessedSegment, system_prompt, is_list=True)
                
                # If we got a valid list of segments, add them to our results
                if isinstance(processed_batch, list) and processed_batch:
                    all_processed.extend(processed_batch)
                    continue  # Success, move to next batch
                
                # If we got a string or empty list, fall back to traditional parsing
                if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
                    st.warning("Pydantic-AI validation returned a string or empty list, falling back to traditional parsing")
            
            except Exception as e:
                if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
                    st.warning(f"Pydantic-AI validation failed: {str(e)}, falling back to traditional parsing")
            
            # Second try: Traditional LLM call with json extraction
            try:
                response = self._call_llm(prompt, system_prompt)
                
                # Try to extract JSON from the response
                json_match = re.search(r'(\[\s*\{.*\}\s*\])', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        processed_batch = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try to fix common JSON issues
                        fixed_json = json_str.replace("'", '"')  # Replace single quotes
                        fixed_json = re.sub(r',\s*(\}|\])', r'\1', fixed_json)  # Remove trailing commas
                        processed_batch = json.loads(fixed_json)
                else:
                    # Try direct parsing
                    cleaned_response = self._clean_response(response)
                    processed_batch = json.loads(cleaned_response)
                
                # Validate each segment manually
                validated_batch = []
                for segment in processed_batch:
                    try:
                        # Set required fields if missing
                        if "type" not in segment:
                            segment["type"] = "dialogue"
                        if "text" not in segment and "dialogue" in segment:
                            segment["text"] = segment.get("dialogue", "") # Use get with default
                        # Ensure text is not None
                        elif segment.get("text") is None:
                            segment["text"] = ""
                        
                        # Ensure speaker consistency
                        if "speaker" in segment and isinstance(segment["speaker"], str):
                            # Clean up speaker name if needed
                            speaker = segment["speaker"].strip()
                            # Make sure audio notations are properly formatted
                            if "(" in speaker and ")" not in speaker:
                                speaker += ")"
                            segment["speaker"] = speaker
                        
                        validated_batch.append(segment)
                    except Exception as e:
                        if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
                            st.warning(f"Segment validation error: {str(e)}")
                        # Use original segment as fallback
                        validated_batch.append(segment)
                
                all_processed.extend(validated_batch)
                
            except Exception as e:
                st.error(f"Failed to parse dialogue processing response: {response[:300]}...")
                st.error(f"Error: {str(e)}")
                # Use original batch on error, but normalize 'text' field first
                normalized_batch = []
                for seg in batch:
                    if seg.get("text") is None:
                        seg["text"] = ""
                    normalized_batch.append(seg)
                all_processed.extend(normalized_batch)
        
        return all_processed