import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def clean_label(label):
    """Clean and format labels for plots"""
    return label.replace("_", " ").title()

def validate_transform(df, col, transform_str):
    """Validate a transformation without applying it to full data"""
    if not transform_str or not transform_str.strip():
        return True, ""
    
    try:
        sample = df[col].head(10).copy()
        x = sample
        allowed_names = {
            "x": x, "np": np, "pd": pd,
            "log": np.log, "log10": np.log10, "log2": np.log2,
            "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
            "sin": np.sin, "cos": np.cos, "tan": np.tan
        }
        result = eval(transform_str, {"__builtins__": {}}, allowed_names)
        
        if not pd.api.types.is_numeric_dtype(result):
            return False, "Result must be numeric"
        
        if np.any(np.isnan(result)):
            return False, "Results in NaN values"
        if np.any(np.isinf(result)):
            return False, "Results in infinite values"
        
        return True, "Valid"
    except Exception as e:
        return False, str(e)[:40]

def apply_transform(df, col, transform_str):
    """Apply mathematical transformation to a column"""
    if not transform_str or not transform_str.strip():
        return df[col].copy()
    
    try:
        x = df[col].copy()
        allowed_names = {
            "x": x, "np": np, "pd": pd,
            "log": np.log, "log10": np.log10, "log2": np.log2,
            "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
            "sin": np.sin, "cos": np.cos, "tan": np.tan
        }
        transformed = eval(transform_str, {"__builtins__": {}}, allowed_names)
        if np.any(np.isinf(transformed)):
            transformed = transformed.replace([np.inf, -np.inf], np.nan)
        return transformed
    except Exception:
        return df[col].copy()

def validate_filter(df, filter_str):
    """Validate a filter expression without applying it"""
    if not filter_str or not filter_str.strip():
        return True, ""
    
    try:
        if any(op in filter_str for op in [">", "<", "==", "!=", "&", "|"]):
            result = df.query(filter_str)
            return True, f"Valid ({len(result)} rows would match)"
        else:
            return False, "Use comparison operators (>, <, ==, etc.)"
    except Exception as e:
        return False, str(e)[:40]

def apply_filter(df, filter_str):
    """Apply filter to DataFrame"""
    if not filter_str or not filter_str.strip():
        return df.copy()
    
    try:
        if any(op in filter_str for op in [">", "<", "==", "!=", "&", "|"]):
            return df.query(filter_str)
        else:
            return df.copy()
    except Exception:
        return df.copy()

def bin_continuous_var(df, col, n_bins):
    """Bin continuous variable for box/violin plots"""
    clean_data = df[col].dropna()
    
    if len(clean_data) == 0:
        return df[col].astype(str)
    
    if clean_data.nunique() <= n_bins:
        return df[col].astype(str)

    bins = n_bins
    labels = None
    
    binned = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    
    if labels is None:
        binned = binned.apply(lambda x: f"{x.left:.3g}-{x.right:.3g}" if pd.notna(x) else "Missing")
    
    return binned

def create_base_viewer(df, stage="base"):
    """Create a generic data visualization dashboard"""
    if df is None or df.empty:
        st.warning("No data available")
        return
    
    if st.button("← Back to Data Upload", type="secondary", key=f"{stage}_back"):
        st.session_state.current_page = 'upload'
        st.rerun()
    
    st.title("Data Analysis")
    
    # Auto-detect column types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Defaults
    default_x = num_cols[0] if num_cols else cat_cols[0] if cat_cols else df.columns[0]
    default_y = num_cols[1] if len(num_cols) > 1 else num_cols[0] if num_cols else None
    default_color = cat_cols[0] if cat_cols else None
    
    # Layout
    col1, col2 = st.columns([1, 3], gap="large")
    
    with col1:
        st.subheader("Controls")
        
        # Plot type
        plot_type = st.selectbox(
            "Plot Type",
            ["scatter", "histogram", "boxplot", "violin", "hexbin"],
            key=f"{stage}_plot_type"
        )
        
        # Axis selectors
        x_axis = st.selectbox(
            "X Axis",
            df.columns.tolist(), 
            index=df.columns.tolist().index(default_x) if default_x in df.columns else 0,
            key=f"{stage}_x_axis"
        )
        
        if num_cols:
            y_axis = st.selectbox(
                "Y Axis",
                num_cols, 
                index=num_cols.index(default_y) if default_y in num_cols else 0,
                key=f"{stage}_y_axis"
            )
        else:
            y_axis = None
            st.warning("No numeric columns found for Y-axis")
        
        # Transform inputs
        x_transform = st.text_input(
            "X Transform", 
            placeholder="np.log10(x) or log(x)",
            key=f"{stage}_x_transform"
        )
        y_transform = st.text_input(
            "Y Transform", 
            placeholder="y*100 or log(y)",
            key=f"{stage}_y_transform"
        )
        
        # Encoding options
        all_cols = cat_cols + num_cols
        
        if plot_type != "hexbin":
            color_by = st.selectbox(
                "Color By", 
                [None] + all_cols, 
                index=([None] + all_cols).index(default_color) if default_color in all_cols else 0,
                key=f"{stage}_color"
            )
        else:
            color_by = None
        
        if plot_type == "scatter":
            size_by = st.selectbox(
                "Size By", 
                [None] + num_cols, 
                key=f"{stage}_size"
            )
        else:
            size_by = None
        
        # Visual controls
        if plot_type in ["scatter", "histogram", "boxplot", "violin"]:
            opacity = st.slider(
                "Opacity", 
                min_value=0.1, max_value=1.0, value=0.7, step=0.1, 
                key=f"{stage}_opacity"
            )
        else:
            opacity = 0.7

        if plot_type in ["histogram", "hexbin"]:
            n_bins = st.slider(
                "Bins",
                min_value=5, max_value=20, value=10, step=1,
                key=f"{stage}_bins"
            )
        elif plot_type in ["boxplot", "violin"]:
            n_bins = st.slider(
                "Bins (for continuous X)",
                min_value=5, max_value=20, value=10, step=1,
                key=f"{stage}_bins",
                help="Used to bin continuous X variables for categorical display"
            )
        
        # Filter
        st.subheader("Filter")
        filter_text = st.text_input(
            "Filter Expression",
            placeholder="column_name >= 100 & other_column != 50",
            help="Use & for AND, | for OR",
            key=f"{stage}_filter"
        )
        
        # Validate filter
        if filter_text:
            is_valid, message = validate_filter(df, filter_text)
            if is_valid:
                st.success(message)
            else:
                st.error(message)
    
    with col2:
        st.subheader("Visualization")
        
        if not y_axis and plot_type not in ["histogram"]:
            st.error("Y axis required for this plot type")
            return
        
        # Apply filter
        filtered_df = apply_filter(df, filter_text)
        if filtered_df.empty:
            st.warning("No data after filtering")
            return
        
        # Apply transforms
        df_plot = filtered_df.copy()
        
        if x_transform:
            try:
                df_plot['x_transformed'] = apply_transform(filtered_df, x_axis, x_transform)
                x_col = 'x_transformed'
                x_label = f"{x_transform.replace('x', x_axis)}"
            except:
                x_col = x_axis
                x_label = x_axis
                st.warning(f"X transform failed: {x_transform}")
        else:
            x_col = x_axis
            x_label = x_axis
        
        if y_transform and y_axis:
            try:
                df_plot['y_transformed'] = apply_transform(filtered_df, y_axis, y_transform)
                y_col = 'y_transformed'
                y_label = f"{y_transform.replace('x', y_axis)}"
            except:
                y_col = y_axis
                y_label = y_axis
                st.warning(f"Y transform failed: {y_transform}")
        else:
            y_col = y_axis
            y_label = y_axis
        
        # Create plot
        fig = None
        
        if plot_type == "scatter":
            fig = px.scatter(
                df_plot, x=x_col, y=y_col,
                color=color_by, size=size_by,
                opacity=opacity,
                title=f"{clean_label(y_label) if y_label else 'Data'} vs {clean_label(x_label)}"
            )
        elif plot_type == "histogram":
            target_col = y_col if y_col else x_col
            fig = px.histogram(
                df_plot, x=target_col, color=color_by,
                nbins=n_bins,
                opacity=opacity,
                title=f"Distribution of {clean_label(target_col)}"
            )
        elif plot_type == "boxplot":
            if pd.api.types.is_numeric_dtype(df_plot[x_col]):
                df_plot['x_binned'] = bin_continuous_var(df_plot, x_col, n_bins)
                x_col_for_plot = 'x_binned'
            else:
                x_col_for_plot = x_col
            
            fig = px.box(
                df_plot, x=x_col_for_plot, y=y_col, color=color_by,
                title=f"{clean_label(y_label)} by {clean_label(x_label)}"
            )
        elif plot_type == "violin":
            if pd.api.types.is_numeric_dtype(df_plot[x_col]):
                df_plot['x_binned'] = bin_continuous_var(df_plot, x_col, n_bins)
                x_col_for_plot = 'x_binned'
            else:
                x_col_for_plot = x_col
            
            fig = px.violin(
                df_plot, x=x_col_for_plot, y=y_col, color=color_by,
                title=f"{clean_label(y_label)} by {clean_label(x_label)}"
            )
        elif plot_type == "hexbin":
            fig = go.Figure(data=go.Histogram2d(
                x=df_plot[x_col], y=df_plot[y_col],
                colorscale='Blues'
            ))
            fig.update_layout(title=f"{clean_label(y_label)} vs {clean_label(x_label)} (density)")
        
        if fig:
            # Clean up labels
            fig.update_layout(
                xaxis_title=clean_label(x_label),
                yaxis_title=clean_label(y_label) if y_label else "",
                height=600,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("Statistics")
        stats_html = f"<b>Records displayed:</b> {len(filtered_df)}<br>"
        
        if pd.api.types.is_numeric_dtype(filtered_df[x_axis]):
            stats_html += f"<b>X range:</b> [{filtered_df[x_axis].min():.3g}, {filtered_df[x_axis].max():.3g}]<br>"
        if y_axis and pd.api.types.is_numeric_dtype(filtered_df[y_axis]):
            stats_html += f"<b>Y range:</b> [{filtered_df[y_axis].min():.3g}, {filtered_df[y_axis].max():.3g}]<br>"
            stats_html += f"<b>Y mean:</b> {filtered_df[y_axis].mean():.4f}<br>"
        
        st.markdown(stats_html, unsafe_allow_html=True)