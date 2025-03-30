"""
Docling integration agent for initial document preprocessing
"""
import streamlit as st
from typing import Dict, List, Any, Optional
import re

# Import docling based on the documentation at https://docling-project.github.io/docling/reference/document_converter/
try:
    import docling
    from docling import DocumentConverter, Document, LayoutDetector

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    st.warning("Docling package not found. Run 'pip install docling' to enable enhanced document preprocessing.")

class DoclingAgent:
    """Agent for preprocessing documents using the docling package."""
    
    def __init__(self):
        """Initialize the Docling agent with document converter and layout detector."""
        self.document_converter = None
        self.layout_detector = None
        
        if DOCLING_AVAILABLE:
            try:
                # Initialize document converter for screenplay-specific document processing
                self.layout_detector = LayoutDetector()
                self.document_converter = DocumentConverter(layout_detector=self.layout_detector)
                st.success("Docling document converter initialized successfully.")
            except Exception as e:
                st.error(f"Error initializing Docling document converter: {str(e)}")
    
    def preprocess_document(self, text: str) -> Dict:
        """
        Preprocess the screenplay document using docling document converter.
        
        Args:
            text: The raw screenplay text
            
        Returns:
            Dictionary with preprocessed document structure
        """
        if not DOCLING_AVAILABLE or self.document_converter is None:
            st.warning("Docling not available. Skipping preprocessing.")
            return {"raw_text": text, "segments": []}
        
        try:
            # Create a docling Document
            doc = Document(text=text)
            
            # Use layout detector to analyze document structure
            self.layout_detector.detect(doc)
            
            # Convert document to structured format
            converted_doc = self.document_converter.convert(doc)
            
            # Extract screenplay-specific elements
            segments = self._extract_segments(converted_doc)
            
            # Extract entities from the document
            entities = self._extract_entities(converted_doc)
            
            return {
                "raw_text": text,
                "processed_doc": converted_doc,  # The converted docling document object
                "segments": segments,
                "entities": entities
            }
            
        except Exception as e:
            st.error(f"Error in Docling preprocessing: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return {"raw_text": text, "segments": []}
    
    def _extract_segments(self, doc) -> List[Dict]:
        """Extract segments from docling document using the document converter structure."""
        segments = []
        try:
            # Process the document elements based on the converted document structure
            # According to the docling documentation, the converter produces structured elements
            
            # Iterate through the blocks in the document
            for block in doc.blocks:
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
            import traceback
            st.warning(traceback.format_exc())
            # If there's an error, return an empty list
            segments = []
            
        return segments
    

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