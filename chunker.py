import logging
from pathlib import Path
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling.datamodel.document import DsDocument # Keep the import just in case, though direct object is used

_log = logging.getLogger(__name__)

def chunk_document_from_source(source_path: Path) -> list[str]:
    """
    Loads a document from a source file using DocumentConverter,
    chunks it using HybridChunker, and returns a list of serialized chunk texts.

    Args:
        source_path: Path to the input source file (e.g., .docx, .pdf).

    Returns:
        A list of strings, where each string is a serialized chunk.
    """
    _log.info(f"Loading and converting document from {source_path}...")
    try:
        # Use DocumentConverter to get the DsDocument object
        converter = DocumentConverter()
        conv_result = converter.convert(source=source_path)
        if not conv_result or not conv_result.document:
             _log.error(f"Failed to convert document from {source_path}")
             return []
        doc = conv_result.document # Get the actual DsDocument object
    except FileNotFoundError:
        _log.error(f"Error: Source file not found at {source_path}")
        return []
    except Exception as e:
        _log.error(f"Error during document conversion: {e}")
        return []

    _log.info("Initializing HybridChunker...")
    # Using default tokenizer for now, as in the basic example
    chunker = HybridChunker()

    _log.info("Chunking document...")
    try:
        # Chunk the DsDocument object directly
        chunk_iter = chunker.chunk(dl_doc=doc)
    except Exception as e:
        _log.error(f"Error during chunking: {e}")
        return []

    serialized_chunks = []
    _log.info("Serializing chunks...")
    for i, chunk in enumerate(chunk_iter):
        try:
            enriched_text = chunker.serialize(chunk=chunk)
            serialized_chunks.append(enriched_text)
        except Exception as e:
            _log.error(f"Error serializing chunk {i}: {e}")
            continue

    _log.info(f"Successfully generated {len(serialized_chunks)} chunks.")
    return serialized_chunks

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Use the original source document path
    input_source_path = Path("test4.docx")

    if not input_source_path.exists():
        _log.error(f"Input file {input_source_path} does not exist.")
    else:
        chunks = chunk_document_from_source(input_source_path)

        if chunks:
            print(f"\n--- Generated {len(chunks)} Chunks ---")
            # Print the first 5 chunks as an example
            for i, chunk_text in enumerate(chunks[:5]):
                print(f"\n=== Chunk {i} ===")
                # Limit printing for brevity
                print(repr(f'{chunk_text[:500]}...'))
        else:
            print("Chunking process failed or produced no chunks.")
