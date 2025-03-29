"""
Main processor for screenplay analysis that orchestrates the specialized agents
"""
import json
import streamlit as st
import pandas as pd
import os
import re
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

# Import all agents
from agents.correction_agent import CorrectionAgent
from agents.segmentation_agent import DocumentSegmentationAgent
from agents.entity_agent import EntityRecognitionAgent
from agents.dialogue_agent import DialogueProcessingAgent
from agents.docx_export_agent import DocxExportAgent

class ScreenplayProcessor:
    """Main processor that orchestrates the agents to parse and analyze a screenplay."""
    
    def __init__(self, provider: str, model: str, api_key: Optional[str] = None, ollama_url: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.ollama_url = ollama_url
        self.dashboard_callback = None
        
        # Initialize agents
        self.segmentation_agent = DocumentSegmentationAgent(provider, model, api_key, ollama_url)
        self.entity_agent = EntityRecognitionAgent(provider, model, api_key, ollama_url)
        self.dialogue_agent = DialogueProcessingAgent(provider, model, api_key, ollama_url)
        self.correction_agent = CorrectionAgent(provider, model, api_key, ollama_url)
        
        # Initialize the DOCX export agent (doesn't require LLM parameters)
        self.docx_export_agent = DocxExportAgent()
    
    def set_dashboard_callback(self, callback: Callable):
        """Set a callback function for updating the dashboard."""
        self.dashboard_callback = callback
        
    def process_screenplay(self, text: str, chunk_size: int = 4000) -> Dict:
        """Process a screenplay through the full pipeline."""
        with st.status("Processing screenplay...", expanded=True) as status:
            # Calculate total steps for progress tracking
            total_steps = 4  # Segmentation, Entity Recognition, Dialogue Processing, Correction
            current_step = 0
            
            # Step 1: Segment the document
            current_step += 0.25  # Starting
            self._update_dashboard(current_step, total_steps, "Segmenting document...")
            status.update(label="Segmenting document...")
            segments = self.segmentation_agent.segment_document(text, chunk_size)
            
            if not segments:
                st.error("Failed to segment document. Please try again.")
                return {"segments": [], "entities": {}}
            
            current_step += 0.75  # Completed segmentation
            self._update_dashboard(current_step, total_steps, "Segmentation complete", segments)
            
            # Step 2: Identify entities
            current_step += 0.25  # Starting
            self._update_dashboard(current_step, total_steps, "Identifying entities...")
            status.update(label="Identifying entities...")
            entities = self.entity_agent.identify_entities(segments)
            
            current_step += 0.75  # Completed entity recognition
            self._update_dashboard(current_step, total_steps, "Entity recognition complete", segments)
            
            # Step 3: Process dialogue specifically
            current_step += 0.25  # Starting
            self._update_dashboard(current_step, total_steps, "Processing dialogue...")
            status.update(label="Processing dialogue...")
            dialogue_segments = [s for s in segments if s.get("type") == "dialogue" or "speaker" in s]
            processed_dialogue = self.dialogue_agent.process_dialogue(dialogue_segments)
            
            # Update the original segments with processed dialogue
            non_dialogue_segments = [s for s in segments if s.get("type") != "dialogue" and "speaker" not in s]
            updated_segments = non_dialogue_segments + processed_dialogue
            
            current_step += 0.75  # Completed dialogue processing
            self._update_dashboard(current_step, total_steps, "Dialogue processing complete", updated_segments)
            
            # Step 4: Correct any remaining inconsistencies
            current_step += 0.25  # Starting
            self._update_dashboard(current_step, total_steps, "Correcting inconsistencies...")
            status.update(label="Correcting inconsistencies...")
            corrected_segments = self.correction_agent.correct_inconsistencies(updated_segments, entities)
            
            # Log the segment count after processing
            segment_markers = len([s for s in corrected_segments if s.get("type") == "segment_marker" or 
                                  (s.get("timecode") and self._is_segment_marker(s.get("timecode")))])
            st.write(f"Found {segment_markers} segment markers in the screenplay")
            
            current_step += 0.75  # Completed correction
            self._update_dashboard(current_step, total_steps, "Processing complete!", corrected_segments)
            status.update(label="Processing complete!", state="complete")
            
        return {
            "segments": corrected_segments,
            "entities": entities
        }
    
    def _update_dashboard(self, current_step, total_steps, message, segments=None):
        """Update the dashboard if a callback is set."""
        if self.dashboard_callback:
            self.dashboard_callback(current_step, total_steps, message, segments)
    
    def _is_segment_marker(self, timecode: str) -> bool:
        """Helper method to check if a timecode represents a segment marker."""
        if not isinstance(timecode, str):
            return False
            
        return bool(re.search(r'\*?\*?[\d:]+[-]{5,}', timecode) or 
                  (timecode and len(timecode.strip()) >= 4 and '-' in timecode and ':' in timecode))
    
    def export_to_docx(self, result: Dict, output_path: Optional[str] = None, episode_number: Optional[str] = None) -> str:
        """Export the processed screenplay to a formatted DOCX file."""
        try:
            self._update_dashboard(0, 1, "Starting DOCX export...")
            
            # If no output path provided, create one in the temp directory
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Create temp file with correct extension
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                    output_path = temp_file.name
                st.write(f"Created temporary file: {output_path}")
            
            # Get the segments from the result
            segments = result.get("segments", [])
            
            # Check if we found any segments
            if not segments:
                st.error("No segments found to export.")
                return ""
            
            self._update_dashboard(0.2, 1, f"Exporting {len(segments)} segments to DOCX...")
            
            # Use the DOCX export agent to create the document
            docx_path = self.docx_export_agent.export_to_docx(segments, output_path, episode_number)
            
            if not docx_path:
                st.error("Failed to create DOCX file. See errors above for details.")
                return ""
                
            # Check if file exists and has content
            if os.path.exists(docx_path):
                file_size = os.path.getsize(docx_path)
                st.write(f"Created file size: {file_size} bytes")
                if file_size > 0:
                    self._update_dashboard(1, 1, f"Export complete: {docx_path}")
                    st.success(f"Successfully exported to DOCX: {docx_path}")
                else:
                    st.warning(f"File was created but is empty: {docx_path}")
            else:
                st.error(f"File was not created at the expected location: {docx_path}")
                
            return docx_path
        except Exception as e:
            st.error(f"Error in export_to_docx: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return ""
    
    def generate_summary(self, result: Dict) -> Dict:
        """Generate a summary of the screenplay with improved entity counting."""
        segments = result.get("segments", [])
        entities = result.get("entities", {})
        
        # Log the raw data for debugging
        st.write(f"Processing {len(segments)} segments for summary")
        if len(segments) > 0:
            st.write(f"First segment type: {segments[0].get('type', 'unknown')}")
        
        # Count segment types
        segment_types = {}
        for segment in segments:
            segment_type = segment.get("type", "unknown")
            segment_types[segment_type] = segment_types.get(segment_type, 0) + 1
        
        st.write(f"Segment types found: {list(segment_types.keys())}")
        
        # Count dialogue by character
        character_dialogue = {}
        for segment in segments:
            # Count for formal dialogue segments
            if segment.get("type") == "dialogue":
                character = segment.get("character", "")
                if character:
                    character_dialogue[character] = character_dialogue.get(character, 0) + 1
            
            # Also count segments with speaker field
            elif "speaker" in segment:
                speaker = segment.get("speaker", "")
                # Remove audio notation for counting
                if isinstance(speaker, str):
                    speaker = re.sub(r'\([^)]*\)', '', speaker).strip()
                    if speaker:
                        character_dialogue[speaker] = character_dialogue.get(speaker, 0) + 1
        
        st.write(f"Found {len(character_dialogue)} characters with dialogue")
        
        # Identify scene changes
        scenes = [s for s in segments if s.get("type") == "scene_header" or (
            s.get("text", "").upper().startswith("INT") or 
            s.get("text", "").upper().startswith("EXT")
        )]
        
        st.write(f"Found {len(scenes)} scene headers")
        
        # Count segment markers - more inclusive regex pattern
        segment_markers = []
        for s in segments:
            # Check if it's explicitly marked as a segment marker
            if s.get("type") == "segment_marker":
                segment_markers.append(s)
            # Check if it has a timecode with dashes pattern
            elif "timecode" in s and isinstance(s.get("timecode"), str) and re.search(r'[\d:]+[-]{5,}', s.get("timecode", "")):
                segment_markers.append(s)
        
        st.write(f"Found {len(segment_markers)} segment markers")
        
        # Use either the entities list or directly check segments if needed
        character_count = len(entities.get("characters", []))
        if character_count == 0 and character_dialogue:
            character_count = len(character_dialogue)
        
        location_count = len(entities.get("locations", []))
        if location_count == 0:
            # Extract locations from scene headers
            locations = set()
            for scene in scenes:
                text = scene.get("text", "")
                if isinstance(text, str):
                    # Try to extract location after INT/EXT
                    loc_match = re.search(r'(?:INT|EXT)\.?\s*[-–—]?\s*(.*?)(?:\s*[-–—]\s*|$)', text, re.IGNORECASE)
                    if loc_match:
                        location = loc_match.group(1).strip()
                        if location:
                            locations.add(location)
            location_count = len(locations)
        
        return {
            "segment_counts": segment_types,
            "character_dialogue_counts": character_dialogue,
            "scene_count": len(scenes),
            "character_count": character_count,
            "location_count": location_count,
            "segment_marker_count": len(segment_markers)
        }
        
    def export_json(self, result: Dict) -> str:
        """Export the screenplay analysis to JSON."""
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def export_csv(self, result: Dict) -> Dict[str, pd.DataFrame]:
        """Export the screenplay analysis to CSV dataframes."""
        segments = result.get("segments", [])
        
        # Convert segments to dataframes based on type
        dataframes = {}
        
        # Dialogue dataframe
        dialogue_segments = [s for s in segments if s.get("type") == "dialogue" or "speaker" in s]
        if dialogue_segments:
            dialogue_df = pd.DataFrame(dialogue_segments)
            dataframes["dialogue"] = dialogue_df
            
        # Scene header dataframe
        scene_segments = [s for s in segments if s.get("type") == "scene_header" or (
            isinstance(s.get("text", ""), str) and (
                s.get("text", "").upper().startswith("INT") or 
                s.get("text", "").upper().startswith("EXT")
            )
        )]
        if scene_segments:
            scene_df = pd.DataFrame(scene_segments)
            dataframes["scenes"] = scene_df
            
        # Title dataframe
        title_segments = [s for s in segments if s.get("type") == "title" or (
            isinstance(s.get("text", ""), str) and 
            s.get("text", "").upper().startswith("TITULOK")
        )]
        if title_segments:
            title_df = pd.DataFrame(title_segments)
            dataframes["titles"] = title_df
            
        # Segment markers dataframe
        segment_markers = [s for s in segments if s.get("type") == "segment_marker" or (
            "timecode" in s and isinstance(s.get("timecode"), str) and re.search(r'[\d:]+[-]{5,}', s.get("timecode", ""))
        )]
        if segment_markers:
            markers_df = pd.DataFrame(segment_markers)
            dataframes["segments"] = markers_df
            
        return dataframes