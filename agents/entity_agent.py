"""
Entity Recognition Agent for screenplay parsing
"""
import re
import json
import streamlit as st
from typing import Dict, List, Any

from agents.llm_agent import LLMAgent

class EntityRecognitionAgent(LLMAgent):
    """Agent for identifying and normalizing entities in the screenplay."""
    
    def identify_entities(self, segments: List[Dict]) -> Dict:
        """Identify characters, locations, and other entities in the segments."""
        # First, try to extract entities directly from segments as a fallback
        characters_direct = self._extract_characters_directly(segments)
        locations_direct = self._extract_locations_directly(segments)
        
        # Log what we found directly
        st.write(f"Direct extraction found {len(characters_direct)} characters and {len(locations_direct)} locations")
        
        system_prompt = """
        You are an expert screenplay analyzer. Your task is to extract and normalize all entities from a screenplay.
        Focus on:
        1. Characters - Find all character names and normalize any inconsistencies
        2. Locations - Extract all locations mentioned
        3. Audio notations - Identify all audio notation types (MO, VO, zMO, etc.) and explain their meaning
        
        Format your response as JSON with these keys:
        - "characters": List of unique character names
        - "locations": List of unique locations
        - "audio_notations": Dictionary mapping audio notation to its meaning
        """
        
        # Extract dialogue segments to analyze characters
        dialogue_segments = [s for s in segments if s.get("type") == "dialogue" or "speaker" in s]
        
        # Extract scene headers to analyze locations
        scene_segments = [s for s in segments if s.get("type") == "scene_header"]
        
        prompt = f"""
        Analyze these screenplay segments and identify all entities.
        
        DIALOGUE SEGMENTS:
        ```
        {json.dumps(dialogue_segments, indent=2)}
        ```
        
        SCENE SEGMENTS:
        ```
        {json.dumps(scene_segments, indent=2)}
        ```
        
        Return ONLY the JSON without any additional explanation.
        """
        
        response = self._call_llm(prompt, system_prompt)
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                entities = json.loads(json_str)
            else:
                entities = json.loads(response)
            
            # Merge with directly extracted entities if LLM results are empty or sparse
            if not entities.get("characters") or len(entities.get("characters", [])) < len(characters_direct):
                st.write(f"LLM found {len(entities.get('characters', []))} characters, using direct extraction results instead")
                entities["characters"] = list(characters_direct)
            
            if not entities.get("locations") or len(entities.get("locations", [])) < len(locations_direct):
                st.write(f"LLM found {len(entities.get('locations', []))} locations, using direct extraction results instead")
                entities["locations"] = list(locations_direct)
            
            # Make sure we have audio notations
            if "audio_notations" not in entities:
                entities["audio_notations"] = self._extract_audio_notations_directly(segments)
            
            return entities
        except json.JSONDecodeError:
            st.error(f"Failed to parse entity recognition response as JSON: {response}")
            # Return our directly extracted entities as a fallback
            return {
                "characters": list(characters_direct),
                "locations": list(locations_direct),
                "audio_notations": self._extract_audio_notations_directly(segments)
            }
    
    def _extract_characters_directly(self, segments: List[Dict]) -> set:
        """Extract character names directly from the segments."""
        characters = set()
        
        for segment in segments:
            # Get characters from speaker field
            if "speaker" in segment:
                # Handle case where speaker might be a list
                speaker_data = segment.get("speaker", "")
                if isinstance(speaker_data, list):
                    # Process each speaker in the list
                    for sp in speaker_data:
                        if isinstance(sp, str):
                            # Extract speaker name, removing audio notations
                            clean_speaker = re.sub(r'\([^)]*\)', '', sp).strip()
                            if clean_speaker:
                                characters.add(clean_speaker)
                elif isinstance(speaker_data, str):
                    # Process single speaker string
                    clean_speaker = re.sub(r'\([^)]*\)', '', speaker_data).strip()
                    if clean_speaker:
                        characters.add(clean_speaker)
            
            # Get characters from character field
            if "character" in segment:
                character_data = segment.get("character", "")
                if isinstance(character_data, list):
                    for ch in character_data:
                        if isinstance(ch, str):
                            clean_character = re.sub(r'\([^)]*\)', '', ch).strip()
                            if clean_character:
                                characters.add(clean_character)
                elif isinstance(character_data, str):
                    clean_character = re.sub(r'\([^)]*\)', '', character_data).strip()
                    if clean_character:
                        characters.add(clean_character)
            
            # Check text for potential ALL CAPS character names
            if "text" in segment:
                text = segment.get("text", "")
                if isinstance(text, str):
                    # Find words in ALL CAPS that might be character names (4+ letters)
                    caps_words = re.findall(r'\b[A-Z]{4,}[A-Z]*\b', text)
                    for word in caps_words:
                        if len(word) >= 4 and word not in ["INT", "EXT", "TITLE", "TITULOK"]:
                            characters.add(word)
        
        # Remove any empty strings
        if "" in characters:
            characters.remove("")
            
        return characters
    
    def _extract_locations_directly(self, segments: List[Dict]) -> set:
        """Extract locations directly from scene headers."""
        locations = set()
        
        for segment in segments:
            if segment.get("type") == "scene_header" or (
                "text" in segment and isinstance(segment.get("text", ""), str) and 
                (segment.get("text", "").upper().startswith("INT") or 
                segment.get("text", "").upper().startswith("EXT"))
            ):
                text = segment.get("text", "")
                if isinstance(text, str):
                    # Try to extract location after INT/EXT
                    loc_match = re.search(r'(?:INT|EXT)\.?\s*[-–—]?\s*(.*?)(?:\s*[-–—]\s*|$)', text, re.IGNORECASE)
                    if loc_match:
                        location = loc_match.group(1).strip()
                        if location:
                            locations.add(location)
        
        return locations
    
    def _extract_audio_notations_directly(self, segments: List[Dict]) -> dict:
        """Extract audio notations directly from speaker fields."""
        notations = {}
        
        # Define common audio notations and their meanings
        common_notations = {
            "VO": "Voice Over - Character is speaking but not visible in the scene",
            "MO": "Monologue - Character's inner thoughts or direct address to audience",
            "zMO": "Slovak notation for Monologue Off-screen",
            "OS": "Off Screen - Character is speaking from outside the visible scene",
            "OOV": "Out of View - Character is speaking but not visible within the frame"
        }
        
        # Look for notation patterns in segments
        for segment in segments:
            if "speaker" in segment:
                speaker_data = segment.get("speaker", "")
                if isinstance(speaker_data, str):
                    # Find notation in parentheses
                    notation_match = re.search(r'\(([^)]+)\)', speaker_data)
                    if notation_match:
                        notation = notation_match.group(1).strip()
                        if notation not in notations:
                            # Use common definition if available, otherwise generic
                            notations[notation] = common_notations.get(notation, f"Audio notation for special delivery")
                elif isinstance(speaker_data, list):
                    for speaker in speaker_data:
                        if isinstance(speaker, str):
                            notation_match = re.search(r'\(([^)]+)\)', speaker)
                            if notation_match:
                                notation = notation_match.group(1).strip()
                                if notation not in notations:
                                    notations[notation] = common_notations.get(notation, f"Audio notation for special delivery")
        
        # If we didn't find any, return some common ones
        if not notations:
            return common_notations
            
        return notations