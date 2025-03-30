"""
Correction Agent for fixing inconsistencies in screenplay parsing
"""
import re
import json
import streamlit as st
from typing import Dict, List, Any

from agents.llm_agent import LLMAgent
from models import ProcessedSegment, Entities

class CorrectionAgent(LLMAgent):
    """Agent for identifying and correcting inconsistencies in the screenplay."""
    
    def correct_inconsistencies(self, segments: List[Dict], entities: Dict) -> List[Dict]:
        """Identify and correct inconsistencies in the segments based on entity knowledge using Pydantic-AI."""
        system_prompt = """
        You are an expert screenplay editor. Your task is to identify and correct inconsistencies in a screenplay,
        including typos, formatting errors, and character name variations.
        
        Use the provided entity information to normalize:
        1. Character names - Ensure all references to the same character use a consistent name
        2. Audio notations - Standardize all audio notations to a consistent format
        3. Scene formatting - Ensure scene headers follow a standard format
        
        Return the corrected segments as a JSON array where each object has these fields:
        - type: The segment type (dialogue, scene_header, segment_marker, etc.)
        - speaker: The normalized speaker name (for dialogue)
        - text: The content text
        - timecode: Any timestamp or segment marker
        - Other fields should be preserved
        """
        
        # Process in batches to avoid hitting token limits
        batch_size = 30
        all_corrected = []
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            st.write(f"Correcting inconsistencies batch {i//batch_size + 1}/{(len(segments)-1)//batch_size + 1}...")
            
            prompt = f"""
            Correct inconsistencies in these screenplay segments based on the entity information provided.
            
            ENTITIES:
            ```
            {json.dumps(entities, indent=2)}
            ```
            
            SEGMENTS:
            ```
            {json.dumps(batch, indent=2)}
            ```
            
            Ensure that:
            1. All character names match their canonical form from the entities list
            2. Audio notations are standardized (e.g., always use (VO) not v/o or V.O.)
            3. Scene headers are properly formatted (e.g., "INT. LOCATION - DAY")
            4. Segment markers with timecodes and dashes are preserved exactly
            
            Return a list of corrected segments.
            """
            
            # First try: Use Pydantic-AI for validation
            try:
                # Convert entities to Entities model if not already
                if not isinstance(entities, Entities):
                    try:
                        entities_model = Entities.model_validate(entities)
                    except:
                        # Keep as is if validation fails
                        pass
                
                # Use the call_llm_with_schema method to get validated corrections
                corrected_batch = self._call_llm_with_schema(prompt, ProcessedSegment, system_prompt, is_list=True)
                
                # If we got a valid list of segments, add them to our results
                if isinstance(corrected_batch, list) and corrected_batch:
                    all_corrected.extend(corrected_batch)
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
                        corrected_batch = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try to fix common JSON issues
                        fixed_json = json_str.replace("'", '"')  # Replace single quotes
                        fixed_json = re.sub(r',\s*(\}|\])', r'\1', fixed_json)  # Remove trailing commas
                        corrected_batch = json.loads(fixed_json)
                else:
                    # Try direct parsing
                    cleaned_response = self._clean_response(response)
                    corrected_batch = json.loads(cleaned_response)
                
                # Post-processing to ensure important segments are preserved
                processed_batch = []
                
                # Get canonical character names from entities
                canonical_names = entities.get("characters", [])
                
                for segment in corrected_batch:
                    try:
                        # Special handling for segment markers
                        if "timecode" in segment and isinstance(segment["timecode"], str) and re.search(r'\d+:\d+[-]{5,}', segment["timecode"]):
                            if "type" not in segment or segment["type"] not in ["segment_marker"]:
                                segment["type"] = "segment_marker"
                        
                        # Normalize character names if possible
                        if "speaker" in segment and canonical_names:
                            speaker = segment["speaker"]
                            # Split out any audio notation
                            speaker_name = re.sub(r'\([^)]*\)', '', speaker).strip()
                            audio_notation = ""
                            match = re.search(r'(\([^)]*\))', speaker)
                            if match:
                                audio_notation = match.group(1)
                            
                            # Find closest match in canonical names
                            closest_match = None
                            for name in canonical_names:
                                if name.upper() == speaker_name.upper():
                                    closest_match = name
                                    break
                            
                            # If we found a match, use it
                            if closest_match:
                                segment["speaker"] = closest_match + (" " + audio_notation if audio_notation else "")
                        
                        processed_batch.append(segment)
                    except Exception as e:
                        if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
                            st.warning(f"Segment correction error: {str(e)}")
                        # Use original segment as fallback
                        processed_batch.append(segment)
                
                all_corrected.extend(processed_batch)
                
            except Exception as e:
                st.error(f"Failed to parse correction response: {str(e)}")
                # Use original batch on error
                all_corrected.extend(batch)
        
        return all_corrected