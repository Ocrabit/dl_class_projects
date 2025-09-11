import streamlit as st
import pandas as pd
import os
from pathlib import Path

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def get_sample_data_files():
    """Get list of sample data files from example_data folder"""
    example_data_path = Path(__file__).parent.parent / "example_data"
    if example_data_path.exists():
        csv_files = list(example_data_path.glob("*.csv"))
        return [(f.name, f) for f in csv_files]
    return []

def drop_in_page(df=None):
    st.set_page_config(page_title="data drop-in", layout="wide")
    st.title("data drop-in")

    if df is None:
        # Create two columns for upload and sample data options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Your Data")
            uploaded_file = st.file_uploader("Drop CSV file", type=['csv'])
            if uploaded_file:
                try:
                    df = load_csv(uploaded_file)
                    st.success(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return None
        
        with col2:
            st.subheader("Choose from Sample Data")
            sample_files = get_sample_data_files()
            
            if sample_files:
                sample_names = [name for name, path in sample_files]
                selected_sample = st.selectbox(
                    "Select sample dataset:",
                    [None] + sample_names,
                    format_func=lambda x: "Choose a sample..." if x is None else x,
                    key="sample_data_selector"
                )
                
                if selected_sample:
                    selected_path = next(path for name, path in sample_files if name == selected_sample)
                    try:
                        sample_df = load_csv(selected_path)
                        st.session_state.stored_df = sample_df
                        st.session_state.selected_columns = sample_df.columns.tolist()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading sample: {str(e)}")
                        return None
            else:
                st.info("No sample data found in example_data folder")
        
        if df is None:
            st.info("Upload a CSV file or choose sample data to start")
            return None
    else:
        st.success(f"Using existing data: {len(df)} rows, {len(df.columns)} columns")
        if st.button("Reupload", type="secondary"):
            st.session_state.stored_df = None
            st.rerun()
    
    if df is not None:
        default_selection = st.session_state.get('selected_columns', df.columns.tolist())
        
        selected_columns = st.multiselect(
            "Select columns:",
            options=df.columns.tolist(),
            default=default_selection
        )
        
        if selected_columns:
            filtered_df = df[selected_columns]
            st.write(f"Selected {len(selected_columns)} columns")
            
            preview_rows = st.number_input("Preview rows:", min_value=1, max_value=len(filtered_df), value=10, step=1)
            st.dataframe(filtered_df.head(preview_rows), width='stretch')
            st.write(f"Showing {min(preview_rows, len(filtered_df))} of {len(filtered_df)} rows")
            st.write(f"Total: {len(filtered_df)} rows, {len(filtered_df.columns)} columns")
            
            # Analyze button
            if st.button("Analyze Data", type="primary"):
                return filtered_df
            return None
        else:
            st.warning("Select at least one column")
            return None
    
    return None

if __name__ == "__main__":
    drop_in_page()
