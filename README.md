# Screenplay Parser with LLM Agents

This application uses a multi-agent LLM (Large Language Model) architecture to intelligently parse, analyze, and format screenplay documents, effectively handling challenges like inconsistent formatting, typos, and varied notation styles.

## Features

-   **Comprehensive Parsing:** Processes screenplays from TXT or DOCX files.
-   **Intelligent Entity Recognition:** Identifies and extracts characters, locations, scene details, and dialogue.
-   **Normalization & Correction:** Normalizes inconsistent character names, formatting variations, and recognizes diverse audio notations (e.g., VO, MO, zMO).
-   **Advanced Analysis:** Provides summary statistics, character dialogue distribution, scene breakdowns, and visualizations.
-   **Flexible Export:** Exports results to formatted DOCX, as well as JSON or CSV for raw data access.
-   **User-friendly Interface:** A Streamlit web application provides an intuitive interface for uploading, processing, and exploring screenplay data.

## Project Structure

```
screenplay-parser/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── config.py               # Configuration settings
├── file_utils.py           # File handling utilities
├── processor.py            # Main screenplay processor orchestrating agents
├── agents/                 # Directory for specialized LLM agents
│   ├── llm_agent.py        # Base LLM Agent class
│   ├── segmentation_agent.py # Document segmentation agent
│   ├── entity_agent.py     # Entity recognition agent
│   ├── dialogue_agent.py   # Dialogue processing agent
│   ├── correction_agent.py # Inconsistency correction agent
│   └── docx_export_agent.py  # DOCX formatting and export agent
├── models.py               # Pydantic models for data structures
├── output/                 # Default directory for exported files
├── docker/                 # Docker-related files
│   ├── Dockerfile
│   └── docker_compose.yaml
├── setup/                  # Setup scripts
│   └── setup_ollama.sh     # Script for setting up local Ollama
└── README.md               # This documentation file
```

## How It Works

The system employs a pipeline of specialized LLM agents:

1.  **Document Segmentation Agent**: Splits the screenplay into logical components (scenes, dialogue, actions, segment markers).
2.  **Entity Recognition Agent**: Identifies key entities such as characters, locations, and specific notation styles within the segmented text.
3.  **Dialogue Processing Agent**: Normalizes character names and dialogue formatting for consistency.
4.  **Correction Agent**: Reviews and corrects inconsistencies across the document, ensuring uniformity.
5.  **DOCX Export Agent**: Creates a professionally formatted DOCX document with appropriate styling for segment markers, scene headers, speaker names, and dialogue.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/screenplay-parser.git
    cd screenplay-parser
    ```

2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit app:

```bash
streamlit run main.py
```

Access the application through your web browser as indicated by the Streamlit output.

### LLM Providers

The app supports the following LLM providers, configurable via the sidebar:

1.  **OpenAI** - Requires an API key.
    *   Models: GPT-3.5-Turbo, GPT-4, GPT-4-Turbo
2.  **Ollama** - Requires a local Ollama instance to be running.
    *   Models: Supports various models like Mistral, Llama3.1, Gemma, etc.
3.  **DeepSeek** - Uses DeepSeek models, typically via a local Ollama instance.
    *   Models: DeepSeek Coder, DeepSeek r1 (optimized prompts available).

### Configuration

-   Upload your screenplay document (.txt or .docx).
-   Choose your LLM provider and specific model.
-   Enter your API key if using OpenAI.
-   Adjust processing parameters (e.g., granularity, timeout) as needed in the sidebar.
-   Click "Process Screenplay".
-   Explore the analysis across different tabs (Summary, Characters, Scenes, Dialogue, Export).

## Advanced Configuration

You can adjust processing parameters in the sidebar:

-   **Processing Granularity**: Control the chunk size (number of characters) for LLM processing. Lower values can be more reliable for complex documents but may be slower.
-   **Request Timeout**: Set the maximum time to wait for a response from the LLM for each chunk.
-   **Detailed Progress**: Show more detailed progress information during processing.
-   **Debug Mode**: Display detailed debugging information, including raw LLM responses.

## Setup and Deployment

For local LLM processing and containerized deployment:

-   **Ollama Setup:** The `setup/setup_ollama.sh` script can help install Ollama and pull recommended models on compatible systems.
-   **Docker:** Use the provided `docker/Dockerfile` and `docker/docker_compose.yaml` to build and run the application and an Ollama service in containers. This is useful for creating a consistent, isolated environment.
    ```bash
    docker-compose up --build
    ```

## Example Output

The application provides several tabs for analyzing the parsed screenplay:

-   **Summary**: Key statistics (scene count, character count, etc.) and visualizations of segment types and character dialogue distribution.
-   **Characters**: A list of identified characters and their dialogue line counts.
-   **Scenes**: A breakdown of scenes, often with extracted scene headers or timestamps.
-   **Dialogue**: Samples of dialogue with attributed speakers.
-   **Export**: Options to download the formatted DOCX, or raw data as JSON and CSV.
-   **Raw Data**: A view of the complete parsed data structure.

## License

MIT License