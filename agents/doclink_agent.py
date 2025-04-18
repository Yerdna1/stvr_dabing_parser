"""
Docling integration agent for initial document preprocessing
"""
import logging
import streamlit as st
from typing import Dict, List, Any, Optional
import re
import json
import traceback
import io

# Import necessary docling components based on example
from docling.document import Document
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.input import DocumentConversionInput, DocumentStream

class DoclingAgent:
    """Agent for preprocessing documents using the docling package."""

    def __init__(self):
        """Initialize the Docling agent with document converter and layout detector."""
        self.doc_converter = None
        self.is_available = False # Default to not available

        try:
            # Configure options (assuming PDF input for now, might need adjustment for TXT)
            # The example uses DoclingParseV4DocumentBackend, let's try that.
            # If input is TXT, this might need different options or backend.
            from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend

            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_page_images = False # Don't need images for text processing

            self.doc_converter = DocumentConverter(
                format_options={
                    # Assuming input might be PDF or TXT treated similarly by backend?
                    # Or maybe we need separate converters/options? Start with PDF example.
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options, backend=DoclingParseV4DocumentBackend
                    ),
                     # Add a basic option for TXT as well, maybe it uses the same backend?
                    InputFormat.TXT: PdfFormatOption( # Re-using PdfFormatOption might be wrong, check docling docs if TXT needs specific options
                        pipeline_options=pipeline_options, backend=DoclingParseV4DocumentBackend
                    )
                }
            )
            self.is_available = True
            logging.info("Docling DocumentConverter initialized successfully.")

        except ImportError as ie:
            self.is_available = False
            logging.error(f"Docling import failed during DoclingAgent initialization: {ie}", exc_info=True)
            logging.warning("Docling package might be missing or corrupted. Install/Reinstall with 'pip install docling'.")
        except Exception as e:
            self.is_available = False
            logging.error(f"Error initializing Docling DocumentConverter: {str(e)}", exc_info=True)

    def preprocess_document(self, text: str, filename: str = "input.txt") -> Dict:
        """
        Preprocess the screenplay document using the Docling DocumentConverter.
        
        Args:
            text: The raw screenplay text
            
        Returns:
            Dictionary with preprocessed document structure
        """
        if not self.is_available or self.doc_converter is None:
            st.warning("Docling not available or failed to initialize. Skipping preprocessing.")
            logging.warning("Docling not available or failed to initialize. Skipping preprocessing.")
            return {"raw_text": text, "segments": [], "entities": {}} # Ensure entities key exists

        try:
            logging.info(f"Processing text (length: {len(text)}) with Docling DocumentConverter...")

            # Create input suitable for DocumentConverter (using stream)
            text_stream = io.BytesIO(text.encode('utf-8'))
            doc_stream = DocumentStream(name=filename, stream=text_stream)
            doc_input = DocumentConversionInput.from_streams([doc_stream])

            # Convert the document
            # Use convert_all as per example, even for single doc
            conv_results: List[ConversionResult] = self.doc_converter.convert_all(
                doc_input,
                raises_on_error=False # Get results even if there are errors
            )

            if not conv_results:
                st.error("Docling conversion returned no results.")
                logging.error("Docling conversion returned no results.")
                return {"raw_text": text, "segments": [], "entities": {}}

            conv_res = conv_results[0] # Process the first (only) result

            if conv_res.status == ConversionStatus.SUCCESS:
                logging.info("Docling conversion successful.")
                doc: Document = conv_res.document # Get the processed Document object

                # Extract segments and entities from the Docling Document object
                logging.info("Extracting segments from Docling document...")
                segments = self._extract_segments(doc)
                logging.info("Extracting entities from Docling document...")
                entities = self._extract_entities(doc)

                logging.info(f"Docling preprocessing complete. Found {len(segments)} segments and {len(entities.get('characters', []))} characters.")

                return {
                    "raw_text": text,
                    "processed_doc": doc, # Keep the processed doc object if needed later
                    "segments": segments,
                    "entities": entities
                }
            elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
                 logging.warning(f"Docling conversion partially successful for {filename}. Errors: {conv_res.errors}")
                 st.warning(f"Docling conversion had errors: {conv_res.errors}. Results might be incomplete.")
                 # Still try to extract from the potentially partial document
                 doc: Document = conv_res.document
                 segments = self._extract_segments(doc)
                 entities = self._extract_entities(doc)
                 return {"raw_text": text, "processed_doc": doc, "segments": segments, "entities": entities}
            else: # Failed
                 logging.error(f"Docling conversion failed for {filename}. Status: {conv_res.status}, Errors: {conv_res.errors}")
                 st.error(f"Docling conversion failed: {conv_res.errors}")
                 return {"raw_text": text, "segments": [], "entities": {}}

        except Exception as e:
            st.error(f"Error during Docling preprocessing pipeline: {str(e)}")
            logging.error(f"Error during Docling preprocessing pipeline: {str(e)}", exc_info=True)
            st.error(traceback.format_exc())
            return {"raw_text": text, "segments": [], "entities": {}}

    def _extract_segments(self, doc: Document) -> List[Dict]: # Added type hint
        """Extract segments from docling document using the document converter structure."""
        segments = []
        try:
            
            # Process the document elements based on the converted document structure
            # According to the docling documentation, the converter produces structured elements
            
            logging.info(f"Extracting segments from {len(doc.blocks) if hasattr(doc, 'blocks') else 0} blocks...")
            
            # Iterate through the blocks in the document
            for block in getattr(doc, 'blocks', []):
                # Get the block text
                block_text = block.text if hasattr(block, 'text') else ""
                
                # Skip empty blocks
                if not block_text.strip():
                    continue
                
                # Get block style/type information
                block_style = block.style if hasattr(block, 'style') else None
                block_type = block.type if hasattr(block, 'type') else None
                
                # Process block based on detected layout style
                if block_style == "heading" or (block_type and block_type == "heading"):
                    # Scene headers are often detected as headings
                    if self._is_scene_header(block_text):
                        segments.append({
                            "type": "scene_header",
                            "text": block_text
                        })
                    else:
                        segments.append({
                            "type": "heading",
                            "text": block_text
                        })
                elif self._is_segment_marker(block_text):
                    # Check for segment markers (timecodes with dashes)
                    segments.append({
                        "type": "segment_marker",
                        "timecode": block_text,
                        "text": ""
                    })
                elif self._is_dialogue(block_text):
                    # For dialogue segments
                    speaker, text = self._parse_dialogue(block_text)
                    segments.append({
                        "type": "dialogue",
                        "speaker": speaker,
                        "text": text
                    })
                elif block_style == "character" or (block_type and block_type == "character"):
                    # Character names (often centered and all caps)
                    segments.append({
                        "type": "character",
                        "speaker": block_text,
                        "text": ""
                    })
                elif block_style == "dialogue" or (block_type and block_type == "dialogue"):
                    # Dialogue text
                    # Check if previous segment was a character
                    if segments and segments[-1].get("type") == "character":
                        # Add as dialogue to previous character
                        segments.append({
                            "type": "dialogue",
                            "speaker": segments[-1].get("speaker", ""),
                            "text": block_text
                        })
                    else:
                        # Standalone dialogue
                        segments.append({
                            "type": "dialogue",
                            "speaker": "",
                            "text": block_text
                        })
                else:
                    # For other text
                    segments.append({
                        "type": "text",
                        "text": block_text
                    })
        
        except Exception as e:
            st.warning(f"Error extracting segments with docling: {str(e)}")
            logging.error(f"Error extracting segments with docling: {str(e)}", exc_info=True)
            st.warning(traceback.format_exc())
            # If there's an error, return an empty list
            segments = []
            
        return segments
    
    def _extract_entities(self, doc) -> Dict:
        """Extract entities from the converted docling document."""
        entities = {
            "characters": [],
            "locations": [],
            "audio_notations": {}
        }
        
        try:
            # Extract characters based on document structure
            # According to the documentation, we can analyze block styles/types
            
            # Keep track of common audio notations
            common_notations = {
                "VO": "Voice Over - Character is speaking but not visible in the scene",
                "MO": "Monologue - Character's inner thoughts or direct address to audience",
                "zMO": "Slovak notation for Monologue Off-screen",
                "OS": "Off Screen - Character is speaking from outside the visible scene",
                "OOV": "Out of View - Character is speaking but not visible within the frame"
            }
            
            # First, extract characters from character-style blocks
            for block in getattr(doc, 'blocks', []):
                block_style = block.style if hasattr(block, 'style') else None
                block_type = block.type if hasattr(block, 'type') else None
                block_text = block.text if hasattr(block, 'text') else ""
                
                if block_style == "character" or (block_type and block_type == "character"):
                    # Clean up character name (remove audio notations)
                    character = re.sub(r'\([^)]*\)', '', block_text).strip()
                    if character:
                        entities["characters"].append(character)
                        
                        # Check for audio notations
                        notation_match = re.search(r'\(([^)]+)\)', block_text)
                        if notation_match:
                            notation = notation_match.group(1).strip()
                            if notation not in entities["audio_notations"]:
                                entities["audio_notations"][notation] = common_notations.get(
                                    notation, f"Audio notation for special delivery"
                                )
                
                # Also check for scene headers to extract locations
                if (block_style == "heading" or (block_type and block_type == "heading")) and self._is_scene_header(block_text):
                    # Extract location from scene header
                    location_match = re.search(r'(?:INT|EXT)\.?\s*[-–—]?\s*(.*?)(?:\s*[-–—]\s*|$)', block_text, re.IGNORECASE)
                    if location_match:
                        location = location_match.group(1).strip()
                        if location:
                            entities["locations"].append(location)
            
            # Second extraction method: analyze dialogue segments
            # Find all-caps words that might be character names
            for block in getattr(doc, 'blocks', []):
                text = block.text if hasattr(block, 'text') else ""
                if text:
                    # Find all-caps words that might be character names
                    cap_matches = re.findall(r'\b[A-Z][A-Z\s]+[A-Z]\b', text)
                    for match in cap_matches:
                        # Exclude common words that might be all caps
                        if match not in ["INT", "EXT", "CUT TO", "FADE IN", "FADE OUT", "DISSOLVE TO"]:
                            clean_name = match.strip()
                            if len(clean_name) >= 2:  # Ensure it's not just a single letter
                                entities["characters"].append(clean_name)
            
            # Deduplicate lists
            entities["characters"] = list(set(entities["characters"]))
            entities["locations"] = list(set(entities["locations"]))
            
            # Add common audio notations if none were found
            if not entities["audio_notations"]:
                entities["audio_notations"] = common_notations
                
        except Exception as e:
            st.warning(f"Error extracting entities with docling: {str(e)}")
            logging.error(f"Error extracting entities with docling: {str(e)}", exc_info=True)
            st.warning(traceback.format_exc())
            
        return entities
    
    def _is_segment_marker(self, text: str) -> bool:
        """Determine if text is a segment marker (timecodes with dashes)."""
        # Handle empty or None text
        if not text or not isinstance(text, str):
            return False
            
        # Check for various segment marker patterns
        if re.search(r'\d+:\d+[-]{5,}', text):
            return True
            
        if re.search(r'\*\*\d+:\d+[-]{5,}', text):
            return True
            
        if re.search(r'\d+:\d+:\d+[-]{5,}', text):
            return True
            
        if re.search(r'[A-Z]\s*\d+:\d+[-]{5,}', text):
            return True
            
        # For backward compatibility - old pattern with at least 5 dashes
        if re.search(r'\*?\*?[\d:]+[-]{5,}', text):
            return True
            
        return False
    
    def _is_scene_header(self, text: str) -> bool:
        """Determine if text is a scene header (starts with INT or EXT)."""
        if not text or not isinstance(text, str):
            return False
            
        # Check for scene header patterns
        if text.strip().upper().startswith(("INT", "EXT")):
            return True
            
        # More comprehensive pattern matching
        if re.match(r'^(INT|EXT|INT\.|EXT\.)[\s./-]', text.strip().upper()):
            return True
            
        return False
    
    def _is_dialogue(self, text: str) -> bool:
        """Determine if text appears to be dialogue."""
        if not text or not isinstance(text, str):
            return False
            
        # Check for character name followed by dialogue patterns
        # Pattern 1: All caps followed by colon
        if re.search(r'^[A-Z][A-Z\s,]+(\([^)]+\))?\s*:', text):
            return True
            
        # Pattern 2: All caps followed by dialogue (no colon)
        if re.search(r'^[A-Z][A-Z\s,]+(\([^)]+\))?\s*[^A-Z]', text):
            return True
            
        return False
    
    def _parse_dialogue(self, text: str):
        """Parse dialogue text into speaker and content."""
        if not text or not isinstance(text, str):
            return "", ""
        
        # Pattern 1: Speaker with colon
        match = re.match(r'^([A-Z][A-Z\s,]+(\([^)]+\))?):\s*(.*)', text)
        if match:
            speaker = match.group(1).strip()
            dialogue = match.group(3).strip()
            return speaker, dialogue
        
        # Pattern 2: Speaker without colon
        match = re.match(r'^([A-Z][A-Z\s,]+(\([^)]+\))?)\s+([^A-Z].*)', text)
        if match:
            speaker = match.group(1).strip()
            dialogue = match.group(3).strip()
            return speaker, dialogue
        
        # Fallback
        return "", text
