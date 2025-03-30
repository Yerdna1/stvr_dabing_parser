import logging
from pathlib import Path
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

_log = logging.getLogger(__name__)

def convert_and_chunk(source_path: Path) -> list[str] | None:
    """
    Loads and converts a source document, then chunks it using HybridChunker.

    Args:
        source_path: Path to the input source file (e.g., .docx, .pdf).

    Returns:
        A list of serialized text chunks, or None if an error occurs.
    """
    # 1. Convert document
    _log.info(f"Loading and converting document from {source_path}...")
    try:
        converter = DocumentConverter()
        conv_result = converter.convert(source=source_path)
        if not conv_result or not conv_result.document:
             _log.error(f"Failed to convert document from {source_path}")
             return None
        doc = conv_result.document
    except FileNotFoundError:
        _log.error(f"Error: Source file not found at {source_path}")
        return None
    except Exception as e:
        _log.error(f"Error during document conversion: {e}")
        return None

    # 2. Chunk document
    _log.info("Initializing HybridChunker...")
    try:
        chunker = HybridChunker() # Using default tokenizer
    except Exception as e:
        _log.error(f"Error initializing HybridChunker: {e}")
        return None

    _log.info("Chunking document...")
    try:
        chunk_iter = chunker.chunk(dl_doc=doc)
        serialized_chunks = [chunker.serialize(chunk=chunk) for chunk in chunk_iter]
    except Exception as e:
        _log.error(f"Error during chunking: {e}")
        return None

    _log.info(f"Generated {len(serialized_chunks)} chunks.")
    if not serialized_chunks:
        _log.warning("No chunks were generated.")
        # Return empty list instead of None if conversion succeeded but no chunks made
        return []

    return serialized_chunks

if __name__ == '__main__':
    # Example usage for testing the module directly
    logging.basicConfig(level=logging.INFO)
    test_file = Path("test4.docx")
    if test_file.exists():
        chunks = convert_and_chunk(test_file)
        if chunks is not None:
            print(f"Successfully generated {len(chunks)} chunks.")
            # print("First chunk:", chunks[0][:500] + "..." if chunks else "N/A")
        else:
            print("Chunking failed.")
    else:
        print(f"Test file {test_file} not found.")
