import logging
import re

_log = logging.getLogger(__name__)

# Column Headers - Define here for consistency
COLUMN_HEADERS = ["Segment", "Speaker", "Timecode", "Text", "Scene Marker", "Segment Marker"]

# --- Regular Expression Patterns ---
P_SEGMENT_MARKER_FIND = re.compile(r"[-–—]{5,}")
P_TIMECODE_FIND = re.compile(r"((?:A\s*)?\b\d{2}:\d{2}(?::\d{2})?(?:-\s*\d{2}:\d{2}(?::\d{2})?)?\b)")
P_SCENE_KEYWORD_FIND = re.compile(r"(INT\.|EXT\.|TITULOK)\s*")
P_PARENS_MARKER_FIND = re.compile(r"(\(.*?\))")
P_SPEAKER_MARKER_IN_TEXT = re.compile(r"^\s*(\(.*\))\s*")
P_COMMA_SEPARATOR = re.compile(r"\s*,\s*")
P_SCRIPT_START_MARKER = re.compile(r"^\d{2}:\d{2}|^-{5,}|^A\s*\d{2}:\d{2}")
P_LIKELY_DIALOGUE_SEP = re.compile(r"\t|\s{2,}")

# -- Speaker Patterns --
SPEAKER_BASE = r"[A-ZÁČĎÉÍĹĽŇÓŔŠŤÚÝŽ\s]+\d?"
SPEAKER_PATTERN = rf"({SPEAKER_BASE}):*"

# -- Fallback Speaker Patterns (Used when NO list is extracted OR list match fails) --
SPEAKER_BASE_FALLBACK = r"[A-ZÁČĎÉÍĹĽŇÓŔŠŤÚÝŽ\s]+\d?"
SPEAKER_PATTERN_FALLBACK = rf"({SPEAKER_BASE_FALLBACK}):*"
# 1. Fallback Multi-speaker comma list followed by whitespace/tab and text
P_MULTI_SPEAKER_TEXT_FALLBACK = re.compile(rf"^({SPEAKER_PATTERN_FALLBACK}(?:,\s*{SPEAKER_PATTERN_FALLBACK})+)\s+(.*)")
# 2. Fallback Single speaker with (MARKER) followed by tab and text
P_SPEAKER_MARKER_FALLBACK = re.compile(rf"^{SPEAKER_PATTERN_FALLBACK}\s*(\(.*\))\s*\t(.*)")
# 3. Fallback Single speaker followed by dash/em-dash and text
P_SPEAKER_DASH_FALLBACK = re.compile(rf"^{SPEAKER_PATTERN_FALLBACK}\s*[-–—]\s*(.*)")
# 4. Fallback Single speaker followed by colon and text
P_SPEAKER_COLON_FALLBACK = re.compile(rf"^{SPEAKER_PATTERN_FALLBACK}:\s+(.*)")
# 5. Fallback Single speaker followed by whitespace/tab and text
P_SPEAKER_SIMPLE_FALLBACK = re.compile(rf"^{SPEAKER_PATTERN_FALLBACK}(\s+)(.*)")


def clean_speaker_name(name: str) -> str:
    """Removes trailing colons (1 or 2) and whitespace from a speaker name."""
    name = name.strip()
    if name.endswith("::"):
        return name[:-2].strip()
    elif name.endswith(":"):
        return name[:-1].strip()
    return name


def extract_speaker_list(chunks: list[str]) -> list[str]:
    """
    Extracts speaker list from 'Postavy:' section. Ignores empty lines within list.
    Stops only when a script start marker is found or max lines reached.
    """
    speakers = []
    in_postavy_section = False
    lines_checked = 0
    max_lines_to_check = 500

    for chunk in chunks:
        lines = chunk.splitlines()
        for line in lines:
            lines_checked += 1
            line_strip = line.strip()

            if not in_postavy_section and "Postavy:" in line:
                in_postavy_section = True
                _log.info("Found 'Postavy:' section.")
                continue

            if in_postavy_section:
                if P_SCRIPT_START_MARKER.match(line_strip):
                    _log.info(f"End of 'Postavy:' section detected (script start marker). Found {len(speakers)} speakers.")
                    unique_speakers = sorted(list(set(s.strip() for s in speakers if s.strip())), key=len, reverse=True)
                    _log.info(f"Final extracted speaker list (sorted): {unique_speakers}")
                    return unique_speakers

                if line_strip:
                    if line_strip[0].isupper() and not P_LIKELY_DIALOGUE_SEP.search(line_strip):
                        cleaned_name = line_strip.rstrip(':').strip()
                        if cleaned_name and len(cleaned_name) < 50:
                            speakers.append(cleaned_name)
                            _log.info(f"Added potential speaker: {cleaned_name}")
                        else:
                             _log.debug(f"Skipping potential speaker line (too long or empty after clean): {repr(line_strip)}")
                    else:
                        _log.debug(f"Skipping potential speaker line (doesn't look like name): {repr(line_strip)}")

            if lines_checked > max_lines_to_check:
                _log.warning(f"Stopped searching for 'Postavy:' after {max_lines_to_check} lines.")
                unique_speakers = sorted(list(set(s.strip() for s in speakers if s.strip())), key=len, reverse=True)
                _log.info(f"Final extracted speaker list (sorted): {unique_speakers}")
                return unique_speakers

    if in_postavy_section:
         _log.warning("'Postavy:' section found but end marker not detected before EOF/limit.")
    else:
         _log.warning("'Postavy:' section not found.")

    unique_speakers = sorted(list(set(s.strip() for s in speakers if s.strip())), key=len, reverse=True)
    _log.info(f"Final extracted speaker list (sorted): {unique_speakers}")
    return unique_speakers


def parse_chunks_to_structured_data(chunks: list[str]) -> list[dict[str, str]]:
    """
    Parses lines using hybrid speaker detection (list prioritized, pattern fallback).
    Includes fallback for multi-speaker lines not in list.

    Args:
        chunks: A list of text chunks.

    Returns:
        A list of dictionaries representing rows.
    """
    parsed_rows = []
    segment_marker_count = 0
    current_segment = 0
    speaker_list = extract_speaker_list(chunks)
    use_speaker_list = bool(speaker_list)

    P_MULTI_SPEAKER_PREFIX_LIST = None
    P_SPEAKER_FIND_LIST = None # Define pattern for single speaker list search
    if use_speaker_list:
        _log.info(f"Using speaker list for detection (priority): {speaker_list}")
        speaker_pattern_str = "|".join(re.escape(s) for s in speaker_list)
        P_SPEAKER_FIND_LIST_PART = rf"({speaker_pattern_str}):*"
        P_MULTI_SPEAKER_PREFIX_LIST = re.compile(rf"^({P_SPEAKER_FIND_LIST_PART}(?:,\s*{P_SPEAKER_FIND_LIST_PART})+)\s+(.*)")
        # Compile single speaker pattern from list here
        P_SPEAKER_FIND_LIST = re.compile(rf"(?<!\w)({speaker_pattern_str}):*\b")
    else:
        _log.error("Could not extract speaker list. Relying on fallback patterns.")


    for chunk_idx, chunk in enumerate(chunks):
        _log.debug(f"--- Parsing Chunk {chunk_idx} ---")
        lines = chunk.splitlines()
        for line_idx, line in enumerate(lines):
            original_line = line.strip()
            if not original_line: continue

            is_segment_marker_line = False
            if P_SEGMENT_MARKER_FIND.search(original_line):
                segment_marker_count += 1
                current_segment = segment_marker_count
                is_segment_marker_line = True
                _log.info(f"Line {line_idx}: Found segment marker sequence. Incrementing segment count to: {current_segment}")

            row_data = {header: "" for header in COLUMN_HEADERS}
            row_data["Segment"] = str(current_segment)
            if is_segment_marker_line: row_data["Segment Marker"] = str(current_segment)

            remaining_text = original_line
            found_spans = []
            speakers_found_on_line = []
            speaker_detection_method = "None"
            speaker_span = None
            text_after_speaker = remaining_text

            # 1. Timecodes
            timecodes = []
            for match in P_TIMECODE_FIND.finditer(remaining_text):
                timecodes.append(match.group(1))
                found_spans.append((*match.span(), "Timecode"))
            if timecodes: row_data["Timecode"] = " ".join(timecodes)

            # --- Speaker Detection ---
            text_for_speaker_search = remaining_text
            for start, end, type in sorted(found_spans, key=lambda x: x[0]):
                 if type == "Timecode": text_for_speaker_search = text_for_speaker_search.replace(remaining_text[start:end], " " * (end - start), 1)
            text_for_speaker_search = text_for_speaker_search.lstrip()
            start_offset = len(remaining_text) - len(text_for_speaker_search)

            # 2a. Try List - Multi-speaker
            multi_match_list = None
            if use_speaker_list and P_MULTI_SPEAKER_PREFIX_LIST:
                 multi_match_list = P_MULTI_SPEAKER_PREFIX_LIST.match(text_for_speaker_search)

            if multi_match_list:
                 speaker_list_str = multi_match_list.group(1)
                 text_after_speaker = multi_match_list.group(multi_match_list.lastindex)
                 # Use findall with the specific list pattern to extract speakers accurately
                 if P_SPEAKER_FIND_LIST: # Check if pattern was compiled
                     speakers_found_on_line = [clean_speaker_name(match.group(1)) for match in P_SPEAKER_FIND_LIST.finditer(speaker_list_str)]
                 else: # Fallback split if pattern failed (shouldn't happen)
                     speakers_found_on_line = [clean_speaker_name(s) for s in speaker_list_str.split(',')]

                 speaker_detection_method = "List-Multi"
                 speaker_span = (start_offset, multi_match_list.end(1) + start_offset)
                 _log.debug(f"Line {line_idx}: Found MULTIPLE speakers (List): {speakers_found_on_line}")

            # 2b. Try List - Single Speaker
            elif use_speaker_list:
                 for known_speaker in speaker_list:
                      if text_for_speaker_search.startswith(known_speaker):
                           end_pos = len(known_speaker)
                           if end_pos == len(text_for_speaker_search) or not text_for_speaker_search[end_pos].isalnum():
                                speakers_found_on_line.append(clean_speaker_name(known_speaker)) # Clean name here
                                speaker_detection_method = "List-Single"
                                speaker_span = (start_offset, end_pos + start_offset)
                                text_after_speaker = text_for_speaker_search[end_pos:].lstrip()
                                _log.debug(f"Line {line_idx}: Found SINGLE speaker (List): {known_speaker}")
                                break

            # 2c. Fallback - Multi-speaker Pattern
            if not speakers_found_on_line:
                 multi_match_fallback = P_MULTI_SPEAKER_TEXT_FALLBACK.match(text_for_speaker_search)
                 if multi_match_fallback:
                      speaker_list_str = multi_match_fallback.group(1)
                      text_after_speaker = multi_match_fallback.group(multi_match_fallback.lastindex)
                      temp_speaker_pattern = re.compile(rf"({SPEAKER_BASE_FALLBACK}):*")
                      speakers_found_on_line = [clean_speaker_name(match.group(1)) for match in temp_speaker_pattern.finditer(speaker_list_str)]
                      speaker_detection_method = "Pattern-Multi-Fallback"
                      speaker_span = (start_offset, multi_match_fallback.end(1) + start_offset)
                      _log.debug(f"Line {line_idx}: Found MULTIPLE speakers (Fallback Pattern): {speakers_found_on_line}")

            # 2d. Fallback - Single Speaker Patterns
            if not speakers_found_on_line:
                 single_match = P_SPEAKER_MARKER_FALLBACK.match(text_for_speaker_search)
                 if single_match: speaker_detection_method = "Pattern-Marker"
                 if not single_match:
                      single_match = P_SPEAKER_DASH_FALLBACK.match(text_for_speaker_search)
                      if single_match: speaker_detection_method = "Pattern-Dash"
                 if not single_match:
                      single_match = P_SPEAKER_COLON_FALLBACK.match(text_for_speaker_search)
                      if single_match: speaker_detection_method = "Pattern-Colon"
                 if not single_match:
                      single_match = P_SPEAKER_SIMPLE_FALLBACK.match(text_for_speaker_search)
                      if single_match: speaker_detection_method = "Pattern-Simple"

                 if single_match:
                      speaker_raw = single_match.group(1)
                      text_after_speaker = single_match.group(single_match.lastindex)
                      # *** Call clean_speaker_name ***
                      speaker_name = clean_speaker_name(speaker_raw)
                      if len(speaker_name) < 50 and speaker_name not in ["INT.", "EXT.", "TITULOK"]:
                           speakers_found_on_line.append(speaker_name)
                           speaker_span = (single_match.start(1) + start_offset, single_match.end(1) + start_offset)
                           _log.debug(f"Line {line_idx}: Found SINGLE speaker ({speaker_detection_method}): {speaker_name}")
                      else:
                           speaker_detection_method = "None"
                           text_after_speaker = text_for_speaker_search
                           speaker_span = None

            if speaker_span: found_spans.append((*speaker_span, "Speaker"))

            remaining_text = text_after_speaker.strip()

            # 3. Scene Markers
            scene_markers = []
            temp_text_for_scene = remaining_text
            keyword_match = P_SCENE_KEYWORD_FIND.search(temp_text_for_scene)
            if keyword_match:
                 original_start = original_line.find(keyword_match.group(0))
                 if original_start != -1:
                      overlaps_speaker = any(max(span[0], original_start) < min(span[1], len(original_line)) and span[2] == "Speaker" for span in found_spans)
                      if not overlaps_speaker:
                           marker_text = original_line[original_start:]
                           scene_markers.append(marker_text.strip())
                           found_spans.append((original_start, len(original_line), "Scene Marker"))
                           _log.debug(f"Line {line_idx}: Found Scene Keyword Marker: {marker_text.strip()}")

            for match in P_PARENS_MARKER_FIND.finditer(original_line):
                 is_covered = any(span[0] <= match.start() and span[1] >= match.end() for span in found_spans if span != match.span())
                 if not is_covered:
                      scene_markers.append(match.group(1))
                      found_spans.append((*match.span(), "Scene Marker"))
                      _log.debug(f"Line {line_idx}: Found Parenthetical Marker: {match.group(1)}")

            if scene_markers: row_data["Scene Marker"] = " ".join(sorted(list(set(scene_markers)), key=original_line.find))

            # --- Determine Final Remaining Text ---
            final_text = remaining_text
            if row_data["Scene Marker"]:
                 for marker in scene_markers:
                      if marker.startswith("(") and marker.endswith(")"): final_text = final_text.replace(marker, "")
                 if keyword_match:
                      if final_text.strip() == scene_markers[0]: final_text = ""

            marker_match = P_SPEAKER_MARKER_IN_TEXT.match(final_text.strip())
            if marker_match: final_text = final_text.replace(marker_match.group(1), '', 1).strip()

            row_data["Text"] = final_text.strip()
            _log.debug(f"Line {line_idx}: Assigned Final Text: {repr(row_data['Text'])}")

            # --- Handle Row Output ---
            if len(speakers_found_on_line) > 1:
                 for sp in speakers_found_on_line:
                      multi_row = row_data.copy(); multi_row["Speaker"] = sp; multi_row["Text"] = row_data["Text"]; parsed_rows.append(multi_row)
                 _log.debug(f"Line {line_idx}: Created {len(speakers_found_on_line)} rows for multiple speakers ({speaker_detection_method}).")
            elif len(speakers_found_on_line) == 1:
                 row_data["Speaker"] = speakers_found_on_line[0]
                 if not (is_segment_marker_line and not row_data["Speaker"] and not row_data["Timecode"] and not row_data["Text"] and not row_data["Scene Marker"]): parsed_rows.append(row_data)
            else: # No speaker found
                 if not (is_segment_marker_line and not row_data["Speaker"] and not row_data["Timecode"] and not row_data["Text"] and not row_data["Scene Marker"]): parsed_rows.append(row_data)

    return parsed_rows


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_chunks = [
        "Postavy:\nANDREJ\nEVA\nPETER KOLAR\nEVA MALA\n\nNINA\nJAN\nMARTIN\nPETER\nJOZO\nJAN4\nJUANA DE ARAG\nDE LA PARRA\n\n00:01:33----------\n00:01:33\tANDREJ\t(dychy) Kde si bola?\nEVA\tNebola som doma.\nPETER KOLAR\t00:02:12\tPridem zajtra\nJAN,MARTIN,PETER,JOZO\tNeprideme tam ani my\n----------\nJUANA DE ARAG\tKde si\nJAN4\tNeviem\nDE LA PARRA\tJa som stale doma\nEVA MALA\t00:03:40\tNebolo to mozne"
    ]
    parsed_data = parse_chunks_to_structured_data(test_chunks)
    print("\nParsed Data:")
    import pandas as pd
    if parsed_data:
        df_test = pd.DataFrame(parsed_data, columns=COLUMN_HEADERS)
        print(df_test.to_string())
    else:
        print("No data parsed.")
