import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
"""
Document Segmentation Agent for screenplay parsing with segment marker detection and newline splitting
"""
import re
import json
import streamlit as st
from typing import Dict, List, Any
from agents.llm_agent import LLMAgent
from models import ProcessedSegment, DialogueSegment


class DocumentSegmentationAgent(LLMAgent):
    """Agent for segmenting the document into logical parts."""
    
    def segment_document(self, text: str, chunk_size: int = 500) -> List[Dict]:
        """Segment document by logical patterns like segment markers with enhanced live visualization."""
        # Create placeholder for live updates
        status_placeholder = st.empty()
        
        # Create a container with fixed height for the scrollable table
        table_container = st.container()
        
        # Initialize DataFrame to store processed segments
        import pandas as pd
        live_segments_df = pd.DataFrame(columns=["Type", "Timecode", "Speaker", "Text"])
        
        # Try to identify segment markers (timecodes with multiple dashes)
        segment_markers = re.findall(r'\d+:\d+[-]{9,}|\*\*\d+:\d+[-]{9,}', text)
        
        if segment_markers and len(segment_markers) > 3:
            # If we have segment markers, split by those
            status_placeholder.write(f"Found {len(segment_markers)} segment markers in the document.")
            
            # Create a pattern that matches any of the segment marker formats
            pattern = r'(\d+:\d+[-]{9,}|\*\*\d+:\d+[-]{9,}\*\*|\*\*\d+:\d+[-]{9,}|[A-Z]\s*\d+:\d+[-]{9,})'
            chunks = re.split(pattern, text)
            processed_chunks = []
            
            # Recombine the splits to keep the markers with their content
            for i in range(0, len(chunks)-1, 2):
                if i+1 < len(chunks):
                    marker = chunks[i+1] if i+1 < len(chunks) else ""
                    content = chunks[i+2] if i+2 < len(chunks) else ""
                    processed_chunks.append(marker + content)
            
            # Don't forget the first chunk if it exists
            if chunks[0]:
                processed_chunks.insert(0, chunks[0])
        else:
            # Fall back to newline-based chunking instead of just character-based
            status_placeholder.write("No segment markers found. Using newline and character-based chunking.")
            
            # First split by newlines to preserve line structure
            lines = text.split('\n')
            
            # Then group lines into chunks of reasonable size to avoid too many small API calls
            processed_chunks = []
            current_chunk = []
            current_size = 0
            
            for line in lines:
                line_size = len(line) + 1  # +1 for the newline
                
                # If adding this line exceeds chunk_size and we already have content, 
                # finalize current chunk and start a new one
                if current_size + line_size > chunk_size and current_chunk:
                    processed_chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Add the line to the current chunk
                current_chunk.append(line)
                current_size += line_size
            
            # Add the last chunk if there's anything left
            if current_chunk:
                processed_chunks.append('\n'.join(current_chunk))
        
        all_segments = []
        segment_count = 0  # Initialize segment counter
        
        # Display the live parsing table with a fixed height container to make it scrollable
        with table_container:
            # Create a container with fixed height (300px) and scrolling
            st.markdown("""
            <style>
            .scrollable-table {
                height: 500px;
                overflow-y: auto;
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create a div with the scrollable class that will contain our table
            st.markdown('<div class="scrollable-table">', unsafe_allow_html=True)
            segments_table = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
        
        for i, chunk in enumerate(processed_chunks):
            status_placeholder.write(f"Processing chunk {i+1}/{len(processed_chunks)}...")
            # Process the chunk and get segments
            segments = self._process_chunk(chunk, i)
            
            # Process segment markers in the results
            processed_segments = []
            for seg in segments:
                # Check if this is a segment marker (timecode with multiple dashes)
                if "timecode" in seg and self._is_segment_marker(seg["timecode"]):
                    segment_count += 1
                    # Add a special segment marker entry
                    new_segment = {
                        "type": "segment_marker",
                        "timecode": seg["timecode"],
                        "segment_number": segment_count,
                        "text": ""  # Can be empty or include any text from the original
                    }
                    processed_segments.append(new_segment)
                    
                    # Add to live visualization
                    new_row = pd.DataFrame([{
                        "Type": "SEGMENT MARKER",
                        "Timecode": seg["timecode"],
                        "Speaker": f"Segment #{segment_count}",
                        "Text": ""
                    }])
                    live_segments_df = pd.concat([live_segments_df, new_row], ignore_index=True)
                else:
                    processed_segments.append(seg)
                    
                    # Add to live visualization
                    segment_type = seg.get("type", "text")
                    if "speaker" in seg:
                        new_row = pd.DataFrame([{
                            "Type": segment_type.upper(),
                            "Timecode": seg.get("timecode", ""),
                            "Speaker": seg.get("speaker", ""),
                            # Get text, ensure it's a string before operating on it
                            "Text": (text_val := seg.get("text", "")), # Get value, default to ""
                            "Text": text_val[:80] + ("..." if len(text_val) > 80 else "") if isinstance(text_val, str) else "" # Process if string
                        }])
                        live_segments_df = pd.concat([live_segments_df, new_row], ignore_index=True)
                    # Ensure text exists, is a string, and is not empty after stripping
                    elif "text" in seg and isinstance(text_val := seg.get("text"), str) and text_val.strip():
                        new_row = pd.DataFrame([{
                            "Type": segment_type.upper(),
                            "Timecode": seg.get("timecode", ""),
                            "Speaker": "",
                            "Text": seg.get("text", "")[:80] + ("..." if len(seg.get("text", "")) > 80 else "")
                        }])
                        live_segments_df = pd.concat([live_segments_df, new_row], ignore_index=True)
                
                # Update the visualization (show up to 50 rows)
                display_df = live_segments_df.tail(50).copy()
                
                # Apply styling to highlight different segment types
                styled_df = display_df.style.apply(
                    lambda x: ['background-color: #ffe0e0' if x['Type'] == 'SEGMENT MARKER' else
                            'background-color: #e0f0ff' if x['Type'] == 'DIALOGUE' else
                            'background-color: #e0ffe0' if x['Type'] == 'SCENE_HEADER' else
                            '' for i in range(len(x))],
                    axis=1
                )
                
                segments_table.dataframe(styled_df, height=500)
            
            all_segments.extend(processed_segments)
        
        status_placeholder.write(f"Processing complete! Found {len(all_segments)} segments.")
        
        # Display final statistics
        segment_markers_count = len([s for s in all_segments if s.get("type") == "segment_marker"])
        speakers_count = len(set([s.get("speaker") for s in all_segments if "speaker" in s]))
        
        st.success(f"✅ Found {segment_markers_count} segment markers and {speakers_count} unique speakers")
        
        return all_segments
    
    def _is_segment_marker(self, timecode: str) -> bool:
        """Determine if a timecode represents a segment marker (contains multiple dashes)."""
        # Handle None values
        if timecode is None:
            return False
            
        # Primary pattern: digits:digits followed by at least 9 dashes
        if re.search(r'\d+:\d+[-]{9,}', timecode):
            return True
            
        # Alternative pattern with asterisks: **digits:digits followed by at least 9 dashes
        if re.search(r'\*\*\d+:\d+[-]{9,}', timecode):
            return True
            
        # Alternative pattern for longer timecodes: digits:digits:digits followed by at least 9 dashes
        if re.search(r'\d+:\d+:\d+[-]{9,}', timecode):
            return True
            
        # Alternative pattern with A, B, C markers: [A-Z]digits:digits followed by at least 9 dashes
        if re.search(r'[A-Z]\s*\d+:\d+[-]{9,}', timecode):
            return True
            
        # For backward compatibility - old pattern with at least 5 dashes
        if re.search(r'\*?\*?[\d:]+[-]{5,}', timecode):
            return True
            
        return False
    
    def _process_chunk(self, text: str, chunk_index: int) -> List[Dict]:
        """Process a single chunk of text with Pydantic-AI validation."""
        # Define the JSON schema for the updated format
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timecode": {"type": "string"},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                    "type": {"type": "string"},
                    "segment_number": {"type": "integer"}
                }
            }
        }
        
        system_prompt = f"""
        You are an expert screenplay parser. Your task is to analyze a screenplay segment and convert it into structured data.
        
        VERY IMPORTANT: Each line in the text (separated by a newline/ENTER) should be treated as a separate segment,
        unless it's clearly part of a longer segment like dialogue.
        
        For each line or segment, extract these components (all are optional):
        1. TIMECODE - Any timestamps in the format like "00:05:44", "1:15:35", "**00:58\\-\\-\\-\\-\\-\\-\\-\\-\\--**", etc.
        2. SPEAKER - Names in ALL CAPS, which may include:
        - Multiple speakers separated by commas (e.g., "PETER, KAROL")
        - Speakers with numbers (e.g., "KAROL1")
        - Audio notation in parentheses like "(VO)", "(MO)", "(zMO)", etc.
        3. TEXT - The actual dialogue or action text
        
        SPECIAL ATTENTION FOR SEGMENT MARKERS:
        - Look for lines with timecodes followed by multiple dashes (at least 5), like "00:05:44----------" or "**06:12\\-\\-\\-\\-\\-\\-\\-\\-\\--**"
        - These are SEGMENT MARKERS and should be preserved exactly as they appear in the "timecode" field
        
        IMPORTANT PARSING RULES:
        - TREAT EACH LINE (separated by newline/ENTER) AS A SEPARATE SEGMENT unless it's clearly part of a continuing dialogue
        - If a line has no clear speaker, put any text content in the "text" field
        - Scene headers, stage directions, and other non-dialogue text should be included in "text"
        - Keep ALL speakers in uppercase as they appear in the original
        - Audio notations like "(VO)" or "(MO)" should be included as part of the speaker field
        - Preserve any segments with timecodes followed by multiple dashes, as these are important structural markers
        """
        
        prompt = f"""
        Analyze this screenplay segment and break it into structured data with timecode, speaker, and text fields.
        This is chunk {chunk_index} of a longer document.
        
        VERY IMPORTANT: 
        - Each line in the text (separated by a newline/ENTER) should generally be treated as a separate segment
        - Only combine lines when they're clearly part of the same dialogue or action
        
        PAY SPECIAL ATTENTION TO SEGMENT MARKERS:
        - These are timecodes followed by multiple dashes (like "00:05:44----------" or "**06:12\\-\\-\\-\\-\\-\\-\\-\\-\\--**")
        - Preserve these exactly as they appear in the "timecode" field
        
        TEXT:
        ```
        {text}
        ```
        
        Return a list of segments, each containing any relevant fields (timecode, speaker, text, type, segment_number).
        For segment markers, the timecode field should contain the exact marker text with dashes.
        For dialogue, include both speaker and text fields.
        For scene headers (e.g. starting with INT or EXT), include in the text field.
        """
        
        # Attempt to process with Pydantic-AI and fallback methods as before...
        # [The rest of the method remains the same]
        
        # First try: Use Pydantic-AI for validation
        try:
            if hasattr(st, 'session_state') and st.session_state.get('detailed_progress', True):
                st.write(f"Processing chunk {chunk_index} with Pydantic-AI")
            
            # Use the _call_llm_with_schema method to get validated segments
            segments = self._call_llm_with_schema(prompt, ProcessedSegment, system_prompt, is_list=True)
            
            # If we got a valid list of segments, normalize and return them
            if isinstance(segments, list) and segments:
                if hasattr(st, 'session_state') and st.session_state.get('detailed_progress', True):
                    st.write(f"Successfully processed {len(segments)} segments with Pydantic-AI")
                return self._normalize_segments(segments)
                
            # If we got a string or empty list, fall back to traditional parsing
            if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
                st.warning("Pydantic-AI validation returned a string or empty list, falling back to traditional parsing")
        
        except Exception as e:
            if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
                st.warning(f"Pydantic-AI validation failed: {str(e)}, falling back to traditional parsing")
        
        # Try simpler line-by-line approach if other methods fail
        lines = text.split("\n")
        if len(lines) > 1 and (not segments or len(segments) < 2):
            try:
                # Process line by line
                line_segments = []
                for line in lines:
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    # Attempt to categorize the line
                    segment = self._categorize_line(line)
                    if segment:
                        line_segments.append(segment)
                
                if line_segments:
                    return self._normalize_segments(line_segments)
            except Exception as e:
                if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
                    st.warning(f"Line-by-line processing failed: {str(e)}")
        
        # Second try: Traditional LLM call with json extraction
        try:
            if hasattr(st, 'session_state') and st.session_state.get('detailed_progress', True):
                st.write(f"Processing chunk {chunk_index} with traditional parsing")
            
            
            response = self._call_llm(prompt, system_prompt)
            
            # Try to parse as JSON directly
            try:
                return self._normalize_segments(segments)
            except json.JSONDecodeError:
                # If direct parsing fails, try extraction techniques
                pass
            
            # Strategy 1: Look for array pattern
            json_match = re.search(r'(\[\s*\{.*\}\s*\])', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    segments = json.loads(json_str)
                    return self._normalize_segments(segments)
                except json.JSONDecodeError:
                    pass
            
            # Strategy 2: Clean and fix common issues
            cleaned_response = re.sub(r'```json|```|<.*?>|\n\s*\n', '', response)
            # Ensure the response starts with [ and ends with ]
            if not cleaned_response.strip().startswith('['):
                cleaned_response = '[' + cleaned_response.strip()
            if not cleaned_response.strip().endswith(']'):
                cleaned_response = cleaned_response.strip() + ']'
                
            try:
                segments = json.loads(cleaned_response)
                return self._normalize_segments(segments)
            except json.JSONDecodeError:
                pass
                
            # Strategy 3: Try to fix incomplete or invalid JSON
            try:
                # Replace single quotes with double quotes
                fixed_response = cleaned_response.replace("'", '"')
                # Fix trailing commas
                fixed_response = re.sub(r',\s*(\}|\])', r'\1', fixed_response)
                # Add missing closing brackets/braces if needed
                open_brackets = fixed_response.count('[')
                close_brackets = fixed_response.count(']')
                if open_brackets > close_brackets:
                    fixed_response += ']' * (open_brackets - close_brackets)
                    
                open_braces = fixed_response.count('{')
                close_braces = fixed_response.count('}')
                if open_braces > close_braces:
                    fixed_response += '}' * (open_braces - close_braces)
                    
                segments = json.loads(fixed_response)
                return self._normalize_segments(segments)
            except json.JSONDecodeError:
                pass
        
        except Exception as e:
            if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
                st.error(f"All parsing attempts failed: {str(e)}")
        
        # Final fallback: simpler prompt with reduced expectations
        try:
            simpler_prompt = f"""
            Parse this screenplay segment into a simple JSON array of objects with these fields:
            - "timecode" (optional): Any timestamps, ESPECIALLY those followed by multiple dashes
            - "speaker" (optional): Names in ALL CAPS (can include multiple speakers)
            - "text" (required): The content or dialogue
            
            IMPORTANT: Each line (separated by newline) should generally be a separate segment.
            
            TEXT:
            ```
            {text[:1000]}  # Shorten the text for the retry
            ```
            
            Return ONLY valid JSON array, no other text.
            """
            
            simpler_response = self._call_llm(simpler_prompt, None)
            
            # Try direct JSON loading
            try:
                segments = json.loads(simpler_response)
                return self._normalize_segments(segments)
            except json.JSONDecodeError:
                # Try one more extraction
                json_match = re.search(r'(\[\s*\{.*\}\s*\])', simpler_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    segments = json.loads(json_str)
                    return self._normalize_segments(segments)
        except:
            pass
        
        # Ultimate fallback: basic line-by-line parsing as separate segments
        lines = text.split('\n')
        basic_segments = []
        for line in lines:
            if line.strip():  # Skip empty lines
                basic_segments.append({"text": line.strip()})
        
        if basic_segments:
            return self._normalize_segments(basic_segments)
        
        # If all else fails, return a single segment with the entire text
        st.error(f"All parsing attempts failed for chunk {chunk_index}. Returning minimal structure.")
        return [{"text": text}]
    
    def _categorize_line(self, line: str) -> Dict:
        """Simple line categorization for fallback processing."""
        line = line.strip()
        if not line:
            return None
            
        # Check for segment markers
        timecode_match = re.search(r'(\d+:\d+[-]{5,}|\*\*\d+:\d+[-]{5,}\*\*|\*\*\d+:\d+[-]{5,})', line)
        if timecode_match:
            return {
                "type": "segment_marker",
                "timecode": timecode_match.group(1),
                "text": ""
            }
            
        # Check for speakers (all caps followed by text)
        speaker_match = re.search(r'^([A-Z][A-Z\s,0-9]+)(\([^)]+\))?\s*[:\-]?\s*(.*)$', line)
        if speaker_match:
            speaker = speaker_match.group(1).strip()
            notation = speaker_match.group(2) or ""
            text = speaker_match.group(3).strip()
            return {
                "type": "dialogue",
                "speaker": (speaker + " " + notation).strip(),
                "text": text
            }
            
        # Check for scene headers
        if line.upper().startswith(('INT', 'EXT')):
            return {
                "type": "scene_header",
                "text": line
            }
            
        # Check for timecodes
        timecode_match = re.search(r'\d+:\d+:\d+|\d+:\d+', line)
        if timecode_match:
            return {
                "type": "text",
                "timecode": timecode_match.group(0),
                "text": line
            }
            
        # Default to plain text
        return {
            "type": "text",
            "text": line
        }
        
    def _normalize_segments(self, segments: List[Dict]) -> List[Dict]:
        """Normalize fields in segments to ensure consistency."""
        normalized = []
        for segment in segments:
            # Handle segment markers first - check for timecodes with dashes
            if "timecode" in segment and self._is_segment_marker(segment["timecode"]):
                # This is a segment marker - keep the timecode field intact
                # Remove any other fields except text
                normalized_segment = {
                    "timecode": segment["timecode"],
                    "text": segment.get("text", "")
                }
                # Add any type field if it exists
                if "type" in segment:
                    normalized_segment["type"] = segment["type"]
                if "segment_number" in segment:
                    normalized_segment["segment_number"] = segment["segment_number"]
                normalized.append(normalized_segment)
                continue
            
            # Handle inconsistent field names
            if "characters" in segment and "speaker" not in segment:
                segment["speaker"] = segment.pop("characters")
            
            if "character" in segment and "speaker" not in segment:
                segment["speaker"] = segment.pop("character")
                
            if "audio_type" in segment and "speaker" in segment:
                # Add audio type to speaker if not already included
                if "(" not in segment["speaker"]:
                    segment["speaker"] = f"{segment['speaker']} ({segment.pop('audio_type')})"
                else:
                    # Remove audio_type as it's already in the speaker
                    segment.pop("audio_type")
                    
            # Ensure type info goes into text field
            if "type" in segment and segment["type"] not in ["dialogue", "speaker", "segment_marker"]:
                if "text" not in segment or not segment["text"]:
                    segment["text"] = f"Type: {segment.pop('type')}"
                elif "scene_type" in segment or "timecode" in segment:
                    type_info = segment.pop("type")
                    scene_type = segment.pop("scene_type", "")
                    time_info = segment.pop("timecode", "")
                    segment["text"] = f"{type_info.upper()} {scene_type} {time_info} {segment.get('text', '')}".strip()
            
            # Remove any old type field unless it's a segment marker
            if "type" in segment and segment["type"] != "segment_marker":
                segment.pop("type")
            
            # Ensure all segments have a text field at minimum
            if "text" not in segment and "speaker" not in segment and "timecode" not in segment:
                continue  # Skip completely empty segments
            
            if "text" not in segment:
                segment["text"] = ""
            # Ensure existing text field is not None
            elif segment.get("text") is None:
                 segment["text"] = ""
            
            normalized.append(segment)
        
        return normalized