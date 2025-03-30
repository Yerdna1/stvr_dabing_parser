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
        """Process a single chunk of text using LLM with fallback to regex."""

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

        # Use simpler LLM call without schema injection
        # Refined system prompt focusing on speaker and scene header extraction
        system_prompt_plain = f"""
        You are an expert screenplay parser. Analyze the provided text chunk and convert it into a JSON array of objects.
        Each object represents a distinct line or segment.

        CRITICAL RULES:
        - Process EACH line separately unless it's clearly continuing dialogue from the previous line.
        - **Speakers:** If a line contains a speaker name (usually ALL CAPS, potentially with notation like (VO)), create an object with a "speaker" field containing the EXACT speaker name (e.g., "JOHN (VO)") and a "text" field containing the dialogue that follows on that line or subsequent lines.
        - **Scene Headers:** If a line starts with INT. or EXT., create an object with the "text" field containing the full scene header line. Add a "type": "scene_header" field.
        - **Segment Markers:** If a line is a segment marker (timecode followed by 5+ dashes like "00:05:44----------"), create an object with the "timecode" field containing the exact marker string and an empty "text" field. Add a "type": "segment_marker" field.
        - **Other Text:** For any other lines (action, description, etc.), create an object with only the "text" field containing the line's content.
        - **Timecodes (Non-Marker):** If a line contains a regular timecode (e.g., "00:01:23") but is NOT a segment marker, include it in the "timecode" field along with the "text".

        OUTPUT FORMAT:
        Return ONLY a valid JSON array. Each object MUST have a "text" field (even if empty for markers). Include "speaker", "timecode", and "type" fields ONLY when identified according to the rules above.

        Example:
        [
          {{ "text": "INT. KITCHEN - DAY", "type": "scene_header" }},
          {{ "text": "Sunlight streams in." }},
          {{ "speaker": "MARTHA (O.S.)", "text": "Coffee's ready!" }},
          {{ "timecode": "00:05:44----------", "text": "", "type": "segment_marker" }},
          {{ "timecode": "00:06:10", "text": "She pours coffee." }}
        ]
        """

        if hasattr(st, 'session_state') and st.session_state.get('detailed_progress', True):
            st.write(f"Processing chunk {chunk_index} with plain LLM call")

        try:
            response = self._call_llm(prompt, system_prompt_plain) # Use plain system prompt
            cleaned_response = self._clean_response(response) # Use robust cleaning

            # Attempt to parse the cleaned response
            segments = self._parse_json_with_retry(cleaned_response) # Use retry parser

            if isinstance(segments, list):
                 if hasattr(st, 'session_state') and st.session_state.get('detailed_progress', True):
                     st.write(f"Successfully parsed {len(segments)} segments from chunk {chunk_index}")
                 return self._normalize_segments(segments)
            else:
                 # If parsing resulted in non-list (e.g., single dict), wrap it
                 logging.warning(f"Parsed non-list result for chunk {chunk_index}, wrapping in list.")
                 return self._normalize_segments([segments])

        except Exception as e:
            st.error(f"LLM processing failed for chunk {chunk_index}: {e}. Falling back to regex categorization.")
            logging.error(f"LLM Error processing chunk {chunk_index}: {e}", exc_info=True)
            # Fallback: Use regex-based line categorization for this chunk
            lines = text.split('\n')
            fallback_segments = []
            for line in lines:
                categorized = self._categorize_line(line)
                if categorized:
                    fallback_segments.append(categorized)

            if fallback_segments:
                 logging.info(f"Using {len(fallback_segments)} segments from regex fallback for chunk {chunk_index}")
                 return self._normalize_segments(fallback_segments) # Normalize the fallback segments
            else:
                 logging.error(f"Regex fallback also failed for chunk {chunk_index}. Returning minimal structure.")
                 return [{"text": text}] # Ultimate fallback if regex also fails

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
            if not isinstance(segment, dict): # Skip non-dict items if any slipped through
                logging.warning(f"Skipping non-dict segment during normalization: {segment}")
                continue

            # Handle segment markers first - check for timecodes with dashes
            timecode = segment.get("timecode")
            if isinstance(timecode, str) and self._is_segment_marker(timecode):
                # This is a segment marker - keep the timecode field intact
                # Ensure 'type' is set correctly
                normalized_segment = {
                    "timecode": timecode,
                    "text": segment.get("text", ""),
                    "type": "segment_marker" # Ensure type is set
                }
                if "segment_number" in segment:
                    normalized_segment["segment_number"] = segment["segment_number"]
                normalized.append(normalized_segment)
                continue

            # Preserve existing speaker if present, otherwise check character(s)
            if "speaker" not in segment or not segment["speaker"]:
                 if "characters" in segment:
                     segment["speaker"] = segment.pop("characters")
                 elif "character" in segment:
                     segment["speaker"] = segment.pop("character")

            # Ensure speaker is a string if it exists
            if "speaker" in segment and not isinstance(segment["speaker"], str):
                 segment["speaker"] = str(segment["speaker"]) # Convert to string

            # Merge audio_type into speaker if present
            if "audio_type" in segment and "speaker" in segment and segment["speaker"]:
                 if "(" not in segment["speaker"]:
                     segment["speaker"] = f"{segment['speaker']} ({segment.pop('audio_type')})"
                 else:
                     segment.pop("audio_type") # Already included

            # Ensure 'text' field exists and is not None
            if "text" not in segment or segment.get("text") is None:
                 segment["text"] = ""

            # Basic check for speaker existence if type is dialogue but speaker is missing
            if segment.get("type") == "dialogue" and not segment.get("speaker"):
                 segment["speaker"] = "" # Add empty speaker field

            # Skip segments that are essentially empty after normalization (unless marker)
            if not segment.get("speaker") and not segment.get("text", "").strip() and not segment.get("timecode"):
                 continue # Skip empty non-marker segments

            normalized.append(segment)

        return normalized
