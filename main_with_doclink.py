"""
Enhanced main.py with docling integration for improved document parsing
"""
import streamlit as st
import os
import json
import time
import pandas as pd
import tempfile
import re
from datetime import datetime

from config import setup_sidebar_config
from file_utils import read_file
from main_with_doclink import EnhancedScreenplayProcessor

# Configure page
st.title("🎬 Enhanced Screenplay Parser with Docling")
st.write("Upload a screenplay document to parse and analyze it using Docling and LLM agents.")

# Setup sidebar configuration
config = setup_sidebar_config()

# Add docling-specific options to sidebar
use_docling = st.sidebar.checkbox("Use Docling for preprocessing", value=True, 
                                help="Enable Docling for initial document parsing")

# Add episode number input for segment numbering
episode_number = st.sidebar.text_input("Episode Number (for segment numbering)", key="episode_number")

# Add a dashboard toggle
show_live_dashboard = st.sidebar.checkbox("Show Live Parsing Dashboard", value=True)

# File upload
uploaded_file = st.file_uploader("Choose a screenplay file", type=["txt", "docx"])

if uploaded_file:
    # Read the file
    text = read_file(uploaded_file)
    
    # Show text preview
    with st.expander("Preview Text"):
        st.text_area("First 1000 characters", text[:1000], height=200)
        st.write(f"Total length: {len(text)} characters")
    
    # Process button
    if st.button("Process Screenplay"):
        # Validate configuration
        if config["llm_provider"] == "OpenAI" and not config["api_key"]:
            st.error("Please enter your OpenAI API key.")
        else:
            # Set up the dashboard if enabled
            if show_live_dashboard:
                dashboard_tabs = st.tabs(["Live Parsing", "Statistics", "Debug Log"])
                
                with dashboard_tabs[0]:
                    # Create placeholders for live updates
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    live_segments = st.empty()
                    segment_count = st.empty()
                    character_count = st.empty()
                
                with dashboard_tabs[1]:
                    # Initialize metrics charts
                    segment_types_chart = st.empty()
                    character_chart = st.empty()
                
                with dashboard_tabs[2]:
                    # Add a debug log area
                    debug_log = st.empty()
                    log_data = []
                    
                    # Function to update debug log
                    def add_to_log(message):
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        log_data.append(f"{timestamp} - {message}")
                        debug_log.code("\n".join(log_data))
                
                # Update initial status
                status_text.write("Initializing processor...")
                add_to_log("Initializing screenplay processor with Docling")
            
            # Initialize enhanced processor with selected provider and docling
            processor = EnhancedScreenplayProcessor(
                provider=config["llm_provider"],
                model=config["model"],
                api_key=config["api_key"] if config["llm_provider"] == "OpenAI" else None,
                ollama_url=config["ollama_url"] if config["llm_provider"] in ["Ollama", "DeepSeek"] else None,
                use_docling=use_docling
            )
            
            # Setup dashboard callbacks if enabled
            if show_live_dashboard:
                # Create a callback function for the processor to update the dashboard
                def update_dashboard(current_step, total_steps, status_message, segments=None):
                    # Update progress bar
                    progress = min(current_step / total_steps, 1.0) if total_steps > 0 else 0
                    progress_bar.progress(progress)
                    
                    # Update status text
                    status_text.write(status_message)
                    
                    # Add to debug log
                    add_to_log(status_message)
                    
                    # Update segment information if available
                    if segments is not None:
                        # Count segment types
                        segment_types = {}
                        for segment in segments:
                            segment_type = segment.get("type", "unknown")
                            segment_types[segment_type] = segment_types.get(segment_type, 0) + 1
                        
                        # Create segment types chart
                        segment_df = pd.DataFrame({
                            "Type": list(segment_types.keys()),
                            "Count": list(segment_types.values())
                        })
                        segment_types_chart.bar_chart(segment_df.set_index("Type"))
                        
                        # Update counts
                        segment_markers = len([s for s in segments if s.get("type") == "segment_marker"])
                        characters = set([s.get("speaker", "") for s in segments if "speaker" in s])
                        
                        segment_count.metric("Segments", segment_markers)
                        character_count.metric("Characters", len(characters))
                        
                        # Display segments with enhanced formatting
                        # Show up to 50 most recent segments
                        display_count = min(50, len(segments))
                        recent_segments = segments[-display_count:] if display_count > 0 else []
                        segments_df = []
                        
                        for seg in recent_segments:
                            if seg.get("type") == "segment_marker":
                                segments_df.append({
                                    "Type": "SEGMENT",
                                    "Timecode": seg.get("timecode", ""),
                                    "Speaker": f"#{seg.get('segment_number', '')}",
                                    "Content": ""
                                })
                            elif "speaker" in seg:
                                segments_df.append({
                                    "Type": seg.get("type", "DIALOGUE"),
                                    "Timecode": seg.get("timecode", ""),
                                    "Speaker": seg.get("speaker", ""),
                                    "Content": seg.get("text", "")[:80] + ("..." if len(seg.get("text", "")) > 80 else "")
                                })
                            else:
                                segments_df.append({
                                    "Type": seg.get("type", "TEXT"),
                                    "Timecode": seg.get("timecode", ""),
                                    "Speaker": "",
                                    "Content": seg.get("text", "")[:80] + ("..." if len(seg.get("text", "")) > 80 else "")
                                })
                        
                        # Create styled dataframe with color highlighting
                        if segments_df:
                            df = pd.DataFrame(segments_df)
                            styled_df = df.style.apply(
                                lambda x: ['background-color: #ffe0e0' if x['Type'] == 'SEGMENT' else
                                        'background-color: #e0f0ff' if x['Type'] == 'DIALOGUE' else
                                        'background-color: #e0ffe0' if x['Type'] == 'SCENE_HEADER' else
                                        '' for i in range(len(x))],
                                axis=1
                            )
                            
                            # Display in scrollable container
                            live_segments.markdown("""
                            <style>
                            .scrollable-container {
                                height: 500px;
                                overflow-y: auto;
                                border: 1px solid #ccc;
                                border-radius: 5px;
                                padding: 10px;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            # Create container for the table
                            live_segments.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
                            live_segments_table = live_segments.empty()
                            live_segments_table.dataframe(styled_df, height=500)
                            live_segments.markdown('</div>', unsafe_allow_html=True)
                        else:
                            live_segments.write("No segments processed yet.")
                
                # Assign the callback to the processor
                processor.set_dashboard_callback(update_dashboard)
            
            # Process the screenplay
            start_time = time.time()
            result = processor.process_screenplay(text, chunk_size=config["parsing_granularity"])
            end_time = time.time()
            
            # If dashboard is enabled, set progress to complete
            if show_live_dashboard:
                progress_bar.progress(1.0)
                status_text.success(f"Processing completed in {end_time - start_time:.2f} seconds!")
                add_to_log(f"Processing complete - total time: {end_time - start_time:.2f} seconds")
            else:
                st.success(f"Processing completed in {end_time - start_time:.2f} seconds!")
            
            # Display the results in tabs
            result_tabs = st.tabs(["Summary", "Characters", "Scenes", "Dialogue", "Export", "Raw Data"])
            
            with result_tabs[0]:  # Summary
                summary = processor.generate_summary(result)
                
                # Display basic stats
                st.subheader("Screenplay Statistics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Scenes", summary["scene_count"])
                col2.metric("Characters", summary["character_count"])
                col3.metric("Locations", summary["location_count"])
                col4.metric("Segments", summary["segment_marker_count"])
                
                # Segment type distribution
                st.subheader("Segment Types")
                segment_df = pd.DataFrame({
                    "Type": list(summary["segment_counts"].keys()),
                    "Count": list(summary["segment_counts"].values())
                })
                st.bar_chart(segment_df.set_index("Type"))
                
                # Character dialogue distribution
                st.subheader("Character Dialogue Distribution")
                if summary["character_dialogue_counts"]:
                    # Sort by count
                    sorted_chars = dict(sorted(
                        summary["character_dialogue_counts"].items(),
                        key=lambda item: item[1],
                        reverse=True
                    ))
                    
                    # Display top 10
                    top_chars = {k: sorted_chars[k] for k in list(sorted_chars.keys())[:10]}
                    char_df = pd.DataFrame({
                        "Character": list(top_chars.keys()),
                        "Lines": list(top_chars.values())
                    })
                    st.bar_chart(char_df.set_index("Character"))
                else:
                    st.write("No character dialogue found.")
            
            with result_tabs[1]:  # Characters
                st.subheader("Characters")
                
                characters = entities = result.get("entities", {}).get("characters", [])
                if characters:
                    # Count dialogue lines per character
                    char_dialogue = {}
                    for seg in result["segments"]:
                        if "speaker" in seg:
                            speaker = seg["speaker"]
                            # Remove audio notation for counting
                            speaker = re.sub(r'\([^)]*\)', '', speaker).strip()
                            if speaker:
                                char_dialogue[speaker] = char_dialogue.get(speaker, 0) + 1
                    
                    # Create character table
                    char_data = []
                    for char in characters:
                        char_data.append({
                            "Character": char,
                            "Dialogue Lines": char_dialogue.get(char, 0)
                        })
                    
                    # Sort by line count
                    char_data = sorted(char_data, key=lambda x: x["Dialogue Lines"], reverse=True)
                    
                    # Display as table
                    st.table(pd.DataFrame(char_data))
                else:
                    st.write("No characters found.")
            
            with result_tabs[2]:  # Scenes
                st.subheader("Scenes")
                
                # Find scene headers
                scene_segments = [s for s in result["segments"] if s.get("type") == "scene_header" or (
                    isinstance(s.get("text", ""), str) and (
                        s.get("text", "").upper().startswith("INT") or 
                        s.get("text", "").upper().startswith("EXT")
                    )
                )]
                
                if scene_segments:
                    # Create scene data
                    scene_data = []
                    for i, scene in enumerate(scene_segments):
                        scene_data.append({
                            "Scene #": i+1,
                            "Type": "INT" if "INT" in scene.get("text", "").upper() else "EXT",
                            "Location": re.sub(r'^(?:INT|EXT)\.?\s*[-–—]?\s*(.*?)(?:\s*[-–—]\s*|$)', 
                                          r'\1', 
                                          scene.get("text", ""), 
                                          flags=re.IGNORECASE).strip(),
                            "Text": scene.get("text", "")
                        })
                    
                    # Display as table
                    st.dataframe(pd.DataFrame(scene_data))
                else:
                    st.write("No scene headers found.")
            
            with result_tabs[3]:  # Dialogue
                st.subheader("Dialogue Samples")
                
                # Find dialogue segments
                dialogue_segments = [s for s in result["segments"] if s.get("type") == "dialogue" or "speaker" in s]
                
                if dialogue_segments:
                    # Take a sample of dialogues (first 20)
                    sample_size = min(20, len(dialogue_segments))
                    sample_dialogues = dialogue_segments[:sample_size]
                    
                    # Create dialogue data
                    dialogue_data = []
                    for i, dlg in enumerate(sample_dialogues):
                        dialogue_data.append({
                            "#": i+1,
                            "Speaker": dlg.get("speaker", ""),
                            "Dialogue": dlg.get("text", "")
                        })
                    
                    # Display as table
                    st.dataframe(pd.DataFrame(dialogue_data))
                    
                    st.write(f"Showing {sample_size} of {len(dialogue_segments)} dialogue segments")
                else:
                    st.write("No dialogue found.")
            
            with result_tabs[4]:  # Export
                st.subheader("Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Export to DOCX")
                    if st.button("Generate DOCX"):
                        with st.spinner("Generating DOCX file..."):
                            # Generate a temporary filename
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"screenplay_{timestamp}.docx"
                            temp_path = os.path.join(tempfile.gettempdir(), filename)
                            
                            # Export to DOCX
                            docx_path = processor.export_to_docx(
                                result, 
                                output_path=temp_path,
                                episode_number=episode_number
                            )
                            
                            if docx_path and os.path.exists(docx_path):
                                # Read the file for download
                                with open(docx_path, "rb") as f:
                                    docx_bytes = f.read()
                                
                                st.download_button(
                                    label="Download DOCX",
                                    data=docx_bytes,
                                    file_name=filename,
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                )
                
                with col2:
                    st.write("Export Raw Data")
                    
                    # Export JSON
                    json_str = processor.export_json(result)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"screenplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                    # Export CSV
                    csv_dataframes = processor.export_csv(result)
                    for name, df in csv_dataframes.items():
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download {name.capitalize()} CSV",
                            data=csv_data,
                            file_name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            with result_tabs[5]:  # Raw Data
                st.subheader("Raw Data")
                
                # Display the raw segments
                with st.expander("Segments (JSON)"):
                    st.json(result["segments"])
                
                # Display the entities
                with st.expander("Entities (JSON)"):
                    st.json(result["entities"])

# Show instructions when no file is uploaded
else:
    st.info("👆 Upload a screenplay file (.txt or .docx) to get started.")
    
    with st.expander("About this enhanced app"):
        st.markdown("""
        ### Enhanced Screenplay Parser with Docling
        
        This app uses the Docling package combined with LLM agents to intelligently parse and analyze screenplay documents.
        
        #### Benefits of using Docling:
        
        - Improved initial document parsing and segmentation
        - Better detection of screenplay elements (characters, scenes, dialogue)
        - Reduced dependency on LLM calls for basic document structure
        - More consistent handling of different document formats
        
        #### How it works
        
        1. **Docling Preprocessing**: Analyzes the document structure using linguistic methods
        2. **Document Segmentation**: Breaks the screenplay into logical parts using Docling results
        3. **Entity Recognition**: Identifies characters, locations, and audio notations
        4. **Dialogue Processing**: Normalizes dialogue and speaker information
        5. **Correction**: Fixes inconsistencies across the document
        6. **DOCX Export**: Creates professionally formatted document with proper styling
        
        #### LLM Providers
        
        You can choose between OpenAI's API (requires API key) or a local Ollama instance
        for processing. For high accuracy, GPT-4 is recommended, but GPT-3.5-Turbo works well too.
        """)

# Footer
st.write("---")
st.caption("Enhanced Screenplay Parser with Docling | Created with Streamlit")