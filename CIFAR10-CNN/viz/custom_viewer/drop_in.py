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

@st.dialog("Sample Data Information")
def show_help_dialog():
    st.markdown("## WandB Sweep Experiment Data")
    
    st.markdown("""
    This data was collected by running **288 hyperparameter sweeps** using Weights & Biases (WandB) 
    across **30 epochs** each.
    """)
    
    st.markdown("### Dataset Descriptions:")
    
    with st.expander("**HistoryOfSweep.csv**", expanded=True):
        st.markdown("""
        - Contains **all epoch states** for each run (epochs 1-30)
        - Includes metrics and values tracked throughout training
        - Larger dataset with detailed training progression
        - Useful for analyzing learning curves and training dynamics
        """)
    
    with st.expander("**SummaryOfSweep.csv**", expanded=True):
        st.markdown("""
        - Contains **final results** from epoch 30 only
        - Includes configuration values used for each run
        - Final metrics and summary statistics
        - Useful for hyperparameter optimization analysis
        """)
    
    st.markdown("### Experiment Details:")
    st.markdown("""
    - **Total Runs:** 288
    - **Epochs per Run:** 30
    - **Data Source:** WandB sweep experiments
    """)
    
    st.markdown("### Sweep Configuration:")
    st.code("""program: train.py
method: grid
metric:
  goal: minimize
  name: val_loss
parameters:
  dropout:
    values: [0.2, 0.3]

  batch_size:
    values: [64, 128]

  learning_rate:
    values: [0.0005, 0.001]

  base_channels:
    values: [32, 64]

  channel_mult:
    values: [2, 3]

  n_conv_layers:
    values: [2, 3, 5]

  apply_dropout_at:
    values: [2]

  activation:
    values: ["ReLU", "GELU", "SiLU"]

  kernel_size:
    values: [3]
  stride:
    values: [2]
  padding:
    values: [1]
  max_channels:
    values: [128]
  seed:
    values: [42]

  epochs:
    value: 30""", language="yaml")
    
    st.markdown("---")
    st.markdown("### Source Code:")
    st.markdown("See the train.py file and full project code at:")
    st.markdown("[https://github.com/Ocrabit/dl_class_projects/tree/main/CIFAR10-CNN](https://github.com/Ocrabit/dl_class_projects/tree/main/CIFAR10-CNN)")
    
    if st.button("Close", type="primary"):
        st.session_state.show_data_help = False
        st.rerun()

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
            col2_1, col2_2, col2_3 = st.columns([10, 2, 1])
            with col2_1:
                st.subheader("Choose from Sample Data")
            with col2_2:
                st.markdown("""
                    <style>
                        .bounce-text {
                            animation: bounce-glow 4s ease-in-out;
                            display: inline-block;
                            font-size: 20px;
                            font-weight: bold;
                        }
                        
                        @keyframes bounce-glow {
                            0% { transform: translateY(0px); text-shadow: none; }
                            8% { transform: translateY(-15px); text-shadow: 0 0 12px rgba(255, 215, 0, 0.9); }
                            16% { transform: translateY(0px); text-shadow: 0 0 6px rgba(255, 215, 0, 0.5); }
                            24% { transform: translateY(-12px); text-shadow: 0 0 10px rgba(255, 215, 0, 0.8); }
                            32% { transform: translateY(0px); text-shadow: 0 0 4px rgba(255, 215, 0, 0.3); }
                            40% { transform: translateY(-8px); text-shadow: 0 0 8px rgba(255, 215, 0, 0.6); }
                            48% { transform: translateY(0px); text-shadow: 0 0 3px rgba(255, 215, 0, 0.2); }
                            56% { transform: translateY(-5px); text-shadow: 0 0 6px rgba(255, 215, 0, 0.4); }
                            64% { transform: translateY(0px); text-shadow: 0 0 2px rgba(255, 215, 0, 0.1); }
                            72% { transform: translateY(-3px); text-shadow: 0 0 4px rgba(255, 215, 0, 0.3); }
                            80% { transform: translateY(0px); text-shadow: 0 0 1px rgba(255, 215, 0, 0.1); }
                            88% { transform: translateY(-1px); text-shadow: 0 0 2px rgba(255, 215, 0, 0.2); }
                            100% { transform: translateY(0px); text-shadow: none; }
                        }
                    </style>
                    <span class="bounce-text">Click me →</span>
                """, unsafe_allow_html=True)
            with col2_3:
                if st.button("?", key="help_button", help="Learn about the sample datasets", width=100):
                    show_help_dialog()
            
            sample_files = get_sample_data_files()
            
            if sample_files:
                sample_names = [name for name, path in sample_files]
                selected_sample = st.selectbox(
                    "Select sample dataset:",
                    [None] + sample_names,
                    format_func=lambda x: "Choose a sample..." if x is None else x,
                    key="sample_data_selector",
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

    st.markdown("""
        <style>
            .github-link {
                font-family: open-sans;
                font-size: 25px;
                font-weight: 500;
                background-color: #eae3de;
                padding: 10px 20px;
                border-radius: 20px;
                text-decoration: none;
                display: inline-block;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .github-link:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                background-color: #94484d;
            }
        </style>
        <div style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); text-align: center;">
            <a href="https://github.com/Ocrabit/dl_class_projects/tree/main/CIFAR10-CNN" target="_blank" class="github-link">
                View project source code on GitHub
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    return None

if __name__ == "__main__":
    drop_in_page()
