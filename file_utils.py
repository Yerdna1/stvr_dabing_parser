"""
File handling utilities for the Screenplay Parser App
"""
import os
import tempfile
import docx2txt
import streamlit as st

def read_file(uploaded_file) -> str:
    """Read content from an uploaded file with proper encoding for Slovak characters."""
    if uploaded_file.name.endswith('.docx'):
        # Create a temporary file to save the uploaded docx
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        # Extract text from the docx file with encoding support
        try:
            text = docx2txt.process(temp_path)
            
            # Check if Slovak characters are present but corrupted
            if any(c in text for c in ['Ä', 'Å', 'Ĺ', 'Ľ', 'Š', 'Č', 'Ť', 'Ž', 'Ý', 'Á', 'Í', 'É']):
                # Try to convert from UTF-8 to correct any encoding issues
                text = text.encode('latin1').decode('utf-8', errors='replace')
        except Exception as e:
            st.warning(f"Warning when reading file: {str(e)}. Trying alternate encoding.")
            # Fallback encoding handling
            text = docx2txt.process(temp_path)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        return text
    elif uploaded_file.name.endswith('.txt'):
        # Try multiple encodings for text files
        try:
            return uploaded_file.getvalue().decode('utf-8')
        except UnicodeDecodeError:
            try:
                return uploaded_file.getvalue().decode('windows-1250')  # Common for Slovak text on Windows
            except UnicodeDecodeError:
                try:
                    return uploaded_file.getvalue().decode('latin2')  # Another encoding for Central European text
                except UnicodeDecodeError:
                    # Final fallback with replacement for invalid chars
                    return uploaded_file.getvalue().decode('utf-8', errors='replace')
    else:
        st.error("Unsupported file format. Please upload a .txt or .docx file.")
        return ""