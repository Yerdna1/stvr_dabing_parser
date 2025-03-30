import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import logging
from io import BytesIO

# Import functions from our modules
from converter import convert_and_chunk
from parser import parse_chunks_to_structured_data, COLUMN_HEADERS # Import headers too

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="Analyzátor Dabingových Scenárov") # Slovak Title

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_log = logging.getLogger(__name__)

# --- Helper Function for Excel Export ---
def to_excel(df: pd.DataFrame) -> bytes:
    """Converts a Pandas DataFrame to an Excel file in memory, ensuring index is the first column named 'Rečník'."""
    output = BytesIO()
    df_reset = df.reset_index()
    if not df_reset.empty:
        original_first_col_name = df_reset.columns[0]
        df_reset = df_reset.rename(columns={original_first_col_name: 'Rečník'})
        _log.info(f"Excel Export: Renamed first column '{original_first_col_name}' to 'Rečník'.")
    else:
        _log.warning("Excel Export: DataFrame is empty, cannot rename index column.")
        if df.index.name:
             df_reset[df.index.name] = []
             df_reset = df_reset.rename(columns={df.index.name: 'Rečník'})

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_reset.to_excel(writer, index=False, sheet_name='Matica_Rečník_Segment')
    processed_data = output.getvalue()
    return processed_data

# --- Main App UI and Logic ---
st.title("🎬 Analyzátor Dabingových Scenárov") # Slovak Title

st.markdown("""
Nahrajte svoj dabingový scenár (ako súbor `.docx`) nižšie.
Aplikácia vykoná nasledovné:
1.  Prekonvertuje a rozdelí dokument na časti (chunks).
2.  Analyzuje text na identifikáciu Rečníkov, Časových kódov, Dialógov, Označení scén a Označení segmentov.
3.  Zobrazí spracované dáta v tabuľke.
4.  Vygeneruje maticu zobrazujúcu, ktorý rečník sa objavuje v ktorom segmente.
5.  Umožní vám stiahnuť maticu Rečník-Segment ako súbor Excel.
""") # Slovak Instructions

uploaded_file = st.file_uploader("Vyberte súbor DOCX", type="docx") # Slovak Label

if uploaded_file is not None:
    st.info(f"Spracováva sa súbor: {uploaded_file.name}") # Slovak Status

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = Path(tmp_file.name)

    try:
        # --- Step 1: Convert and Chunk ---
        with st.spinner("Konvertujem a rozdeľujem dokument..."): # Slovak Spinner
            chunks = convert_and_chunk(tmp_file_path)

        if chunks is None:
            st.error("Nepodarilo sa konvertovať alebo rozdeliť dokument. Skontrolujte logy pre detaily.") # Slovak Error
        elif not chunks:
            st.warning("Dokument bol konvertovaný, ale neboli vygenerované žiadne textové časti (chunks).") # Slovak Warning
        else:
            st.success(f"Dokument úspešne rozdelený na {len(chunks)} častí.") # Slovak Success

            # --- Step 2: Parse Chunks ---
            with st.spinner("Spracovávam časti (chunks)..."): # Slovak Spinner
                parsed_data_raw = parse_chunks_to_structured_data(chunks)

            # --- Clean parsed data: Replace None with '' ---
            parsed_data = []
            if parsed_data_raw:
                for row_dict in parsed_data_raw:
                    cleaned_row = {k: (v if v is not None else '') for k, v in row_dict.items()}
                    parsed_data.append(cleaned_row)
            # --- End Cleaning ---

            if not parsed_data:
                st.warning("Spracovanie dokončené, ale neboli extrahované žiadne štruktúrované dáta.") # Slovak Warning
            else:
                st.success(f"Spracovanie dokončené. Extrahovaných {len(parsed_data)} riadkov.") # Slovak Success

                # --- Step 3: Display Parsed Data Table ---
                st.header("Spracované Dáta Scenára") # Slovak Header
                # Create DataFrame, explicitly setting dtype to str where possible
                try:
                    df_parsed = pd.DataFrame(parsed_data, columns=COLUMN_HEADERS).fillna('').astype(str)
                except Exception as e:
                     _log.error(f"Error creating/casting main DataFrame: {e}. Creating without astype.")
                     # Fallback if astype fails
                     df_parsed = pd.DataFrame(parsed_data, columns=COLUMN_HEADERS).fillna('')

                # Rename columns for display
                df_display = df_parsed.rename(columns={
                    "Segment": "Segment",
                    "Speaker": "Rečník",
                    "Timecode": "Časový kód",
                    "Text": "Text",
                    "Scene Marker": "Označenie Scény",
                    "Segment Marker": "Označenie Segmentu"
                })
                # Display using standard st.dataframe
                st.dataframe(df_display, use_container_width=True)

                # --- Step 4: Generate and Display Speaker-Segment Matrix ---
                st.header("Matica Rečník-Segment") # Slovak Header
                # Ensure 'Segment' column is treated as string for filtering if needed, then filter
                df_parsed_str_segment = df_parsed.copy()
                df_parsed_str_segment['Segment'] = df_parsed_str_segment['Segment'].astype(str)

                df_filtered = df_parsed_str_segment[
                    (df_parsed_str_segment['Speaker'] != '') &
                    (df_parsed_str_segment['Segment'].str.isdigit()) & # Check if string is digit
                    (df_parsed_str_segment['Segment'].astype(int) > 0) # Convert valid ones to int for comparison
                ].copy()

                if not df_filtered.empty:
                    # Convert Segment back to integer for crosstab
                    df_filtered['Segment'] = df_filtered['Segment'].astype(int)
                    try:
                        speaker_matrix = pd.crosstab(
                            index=df_filtered['Speaker'],
                            columns=df_filtered['Segment']
                        )
                        transformed_matrix = speaker_matrix.copy()
                        for segment_col in transformed_matrix.columns:
                            # Ensure segment number is stored as string
                            transformed_matrix[segment_col] = transformed_matrix[segment_col].apply(
                                lambda count: str(segment_col) if count > 0 else ''
                            )

                        # Display matrix - contains only strings now
                        st.dataframe(transformed_matrix, use_container_width=True)

                        # --- Step 5: Prepare Excel Download ---
                        excel_data = to_excel(transformed_matrix)
                        st.download_button(
                            label="📥 Stiahnuť Maticu Rečník-Segment (Excel)", # Slovak Label
                            data=excel_data,
                            file_name=f"{Path(uploaded_file.name).stem}_matica_recnik_segment.xlsx", # Slovak Filename
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Chyba pri vytváraní matice rečníkov: {e}") # Slovak Error
                else:
                    st.warning("Neboli nájdené žiadne dáta rečníkov v platných segmentoch na vytvorenie matice.") # Slovak Warning

    except Exception as e:
        st.error(f"Počas spracovania nastala neočakávaná chyba: {e}") # Slovak Error
        _log.exception("Nespracovaná chyba počas behu aplikácie:") # Log full traceback
    finally:
        # Clean up the temporary file
        if 'tmp_file_path' in locals() and tmp_file_path.exists():
            tmp_file_path.unlink()
            _log.info(f"Dočasný súbor zmazaný: {tmp_file_path}") # Slovak Log

else:
    st.info("Prosím, nahrajte súbor DOCX pre začatie.") # Slovak Info
