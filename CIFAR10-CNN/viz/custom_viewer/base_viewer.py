import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Default maximum point size for scatter plots
DEFAULT_MAX_SIZE = 5

# Maximum number of unique values for hyperparameter filtering
MAX_HYPERPARAM_UNIQUE_VALUES = 15

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
    except (ValueError, TypeError, SyntaxError):
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
    except (ValueError, TypeError, SyntaxError, KeyError):
        return df.copy()

def build_advanced_filter(cat_filter_col, selected_vals, hyperparam_filters):
    """Build filter expression from advanced filter selections"""
    filter_parts = []
    
    # Add categorical filter
    if cat_filter_col and selected_vals:
        conditions = []
        for val in selected_vals:
            if isinstance(val, str):
                escaped_val = val.replace("'", "\\'")  # Escape the damn str
                conditions.append(f"`{cat_filter_col}` == '{escaped_val}'")
            elif pd.isna(val):
                conditions.append(f"`{cat_filter_col}`.isna()")
            else:
                conditions.append(f"`{cat_filter_col}` == {val}")
        
        if conditions:
            filter_parts.append(f"({' | '.join(conditions)})")
    
    # Add hyperparameter filters
    for col, vals in hyperparam_filters.items():
        if vals:
            # Use same approach for hyperparameter filters
            conditions = []
            for val in vals:
                if isinstance(val, str):
                    escaped_val = val.replace("'", "\\'")
                    conditions.append(f"`{col}` == '{escaped_val}'")
                elif pd.isna(val):
                    conditions.append(f"`{col}`.isna()")
                else:
                    conditions.append(f"`{col}` == {val}")
            
            if conditions:
                filter_parts.append(f"({' | '.join(conditions)})")
    
    return " & ".join(filter_parts) if filter_parts else ""

def combine_filters(basic_filter, advanced_filter):
    """Combine basic and advanced filter expressions"""
    if basic_filter and advanced_filter:
        return f"({basic_filter}) & ({advanced_filter})"
    elif basic_filter:
        return basic_filter
    elif advanced_filter:
        return advanced_filter
    else:
        return ""

def apply_groupby(df, group_col, agg_method, numeric_cols):
    """Group DataFrame by specified column and aggregate numeric columns"""
    if not group_col or group_col not in df.columns:
        return df.copy()
    
    try:
        # Group by the specified column
        grouped = df.groupby(group_col)
        
        # Apply aggregation to numeric columns only
        if agg_method == "mean":
            agg_df = grouped[numeric_cols].mean().reset_index()
        elif agg_method == "median":
            agg_df = grouped[numeric_cols].median().reset_index()
        elif agg_method == "sum":
            agg_df = grouped[numeric_cols].sum().reset_index()
        elif agg_method == "count":
            agg_df = grouped[numeric_cols].count().reset_index()
        elif agg_method == "min":
            agg_df = grouped[numeric_cols].min().reset_index()
        elif agg_method == "max":
            agg_df = grouped[numeric_cols].max().reset_index()
        else:  # default to mean
            agg_df = grouped[numeric_cols].mean().reset_index()
        
        # Add categorical columns back (take first value from each group)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [col for col in cat_cols if col != group_col]  # exclude group column
        
        if cat_cols:
            cat_data = grouped[cat_cols].first().reset_index()
            agg_df = agg_df.merge(cat_data, on=group_col, how='left')
        
        return agg_df
        
    except (ValueError, TypeError, KeyError):
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
        # Clear selected columns to avoid mismatch with new data
        if 'selected_columns' in st.session_state:
            del st.session_state.selected_columns
        st.rerun()
    
    st.title("Data Analysis")
    
    # Auto-detect column types (cache these as they're used multiple times)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = cat_cols + num_cols
    
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
        
        # Encoding options in expander
        
        with st.expander("Visual Specifications", expanded=True):
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
                
                if size_by:
                    prev_size_col_key = f"{stage}_prev_size_col"
                    if prev_size_col_key not in st.session_state:
                        st.session_state[prev_size_col_key] = size_by
                    
                    if st.session_state[prev_size_col_key] != size_by:
                        st.session_state[f"{stage}_size_scale"] = DEFAULT_MAX_SIZE
                        st.session_state[prev_size_col_key] = size_by
                
                # Size slider always available for scatter plots
                max_size = st.slider(
                    "Point Size" if not size_by else "Max Point Size",
                    min_value=5,
                    max_value=50,
                    value=int(st.session_state.get(f"{stage}_size_scale", DEFAULT_MAX_SIZE)),
                    step=1,
                    key=f"{stage}_size_scale",
                    help="Size in pixels for points" if not size_by else "Maximum size in pixels for the largest points"
                )
                
            else:
                size_by = None
            
            # Opacity control
            if plot_type in ["scatter", "histogram", "boxplot", "violin"]:
                opacity = st.slider(
                    "Opacity", 
                    min_value=0.1, max_value=1.0, value=0.7, step=0.1, 
                    key=f"{stage}_opacity"
                )
            else:
                opacity = 0.7
        
        # Trendlines feature in expandable section (only for scatter plots)
        if plot_type == "scatter":
            with st.expander("Trendlines", expanded=False):
                show_trendlines = st.checkbox(
                    "Show Trendlines",
                    key=f"{stage}_show_trendlines",
                    help="Connect points into lines based on grouping"
                )
                
                if show_trendlines:
                    trend_by = st.selectbox(
                        "Line Groups",
                        df.columns.tolist(),
                        key=f"{stage}_trend_by",
                        help="Connect points with same value in this column"
                    )
                    
                    # Check if trend column is numeric and needs binning
                    if trend_by and pd.api.types.is_numeric_dtype(df[trend_by]):
                        trend_bins = st.slider(
                            "Groups (bins)",
                            min_value=2,
                            max_value=20,
                            value=5,
                            key=f"{stage}_trend_bins",
                            help="Number of groups to divide continuous data into"
                        )
                    else:
                        trend_bins = None
                    
                    line_opacity = st.slider(
                        "Line Opacity",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.85,
                        step=0.1,
                        key=f"{stage}_line_opacity",
                        help="Opacity of the trendlines"
                    )
                    
                    show_points_with_lines = st.checkbox(
                        "Show Points",
                        value=True,
                        key=f"{stage}_show_points_with_lines",
                        help="Show individual points in addition to trendlines"
                    )
        else:
            show_trendlines = False

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
        
        # Advanced Filter
        with st.expander("Advanced Filter", expanded=False):
            filter_mode = st.radio(
                "Filter Mode",
                ["Categorical Values", "Hyperparameter Combinations"],
                key=f"{stage}_filter_mode",
                help="Choose between categorical filtering or hyperparameter grid filtering"
            )
            
            if filter_mode == "Categorical Values":
                st.markdown("**Select by Categorical Values**")
                
                # Select categorical column to filter
                cat_filter_col = st.selectbox(
                    "Filter by Column",
                    [None] + cat_cols + [col for col in df.columns if col not in num_cols and col not in cat_cols],
                    key=f"{stage}_cat_filter_col",
                    help="Select a categorical column to filter by specific values"
                )
                
                if cat_filter_col:
                    # Get unique values for the selected column
                    unique_vals = df[cat_filter_col].unique().tolist()
                    
                    # Multi-select for values
                    selected_vals = st.multiselect(
                        f"Select {cat_filter_col} values",
                        unique_vals,
                        key=f"{stage}_cat_filter_vals",
                        help="Select one or more values to include"
                    )
            
            else:  # Hyperparameter Combinations
                st.markdown("**Hyperparameter Combinations**")
                st.markdown("Enter in grid style combinations of what you want to select")
                
                # Identify columns that might be hyperparameters (numeric with discrete values)
                hyperparam_cols = [
                    col for col in num_cols
                    if 1 < df[col].nunique() <= MAX_HYPERPARAM_UNIQUE_VALUES
                ]
                
                if hyperparam_cols:
                    st.markdown("Select specific combinations:")
                    
                    for col in hyperparam_cols:  # No limit on number of fields
                        unique_vals = sorted(df[col].unique().tolist())
                        st.multiselect(
                            col,
                            unique_vals,
                            key=f"{stage}_hyper_{col}"
                        )
                else:
                    st.info(f"No discrete numeric columns found for hyperparameter filtering (need >1 and ≤{MAX_HYPERPARAM_UNIQUE_VALUES} unique values)")
        
        # Group By
        st.subheader("Group By")
        group_by_col = st.selectbox(
            "Group By Column",
            [None] + df.columns.tolist(),
            key=f"{stage}_group_by",
            help="Group data by this column and aggregate"
        )
        
        if group_by_col:
            agg_method = st.selectbox(
                "Aggregation Method",
                ["mean", "median", "sum", "count", "min", "max"],
                key=f"{stage}_agg_method",
                help="How to aggregate numeric values within each group"
            )
    
    with col2:
        st.subheader("Visualization")
        
        if not y_axis and plot_type not in ["histogram"]:
            st.error("Y axis required for this plot type")
            return
        
        # Build combined filter from basic and advanced filters
        # Get advanced filter values based on selected mode
        filter_mode = st.session_state.get(f"{stage}_filter_mode", "Categorical Values")
        
        if filter_mode == "Categorical Values":
            cat_filter_col = st.session_state.get(f"{stage}_cat_filter_col", None)
            selected_vals = st.session_state.get(f"{stage}_cat_filter_vals", [])
            hyperparam_filters = {}
        else:
            cat_filter_col = None
            selected_vals = []
            hyperparam_filters = {}
            for key in st.session_state:
                if key.startswith(f"{stage}_hyper_"):
                    col_name = key.replace(f"{stage}_hyper_", "")
                    vals = st.session_state[key]
                    if vals:
                        hyperparam_filters[col_name] = vals
        
        # Build advanced filter expression
        advanced_filter = build_advanced_filter(cat_filter_col, selected_vals, hyperparam_filters)
        
        # Combine with basic filter
        combined_filter = combine_filters(filter_text, advanced_filter)
        
        # Show combined filter if it exists and is non-trivial
        if combined_filter and advanced_filter:
            st.info(f"Combined filter: {combined_filter[:100]}{'...' if len(combined_filter) > 100 else ''}")
        
        # Apply combined filter
        filtered_df = apply_filter(df, combined_filter)
        if filtered_df.empty:
            st.warning("No data after filtering")
            return
        
        # Apply group by if selected
        if group_by_col:
            df_plot = apply_groupby(filtered_df, group_by_col, agg_method, num_cols)
            st.info(f"Grouped by {group_by_col}: {len(df_plot)} groups from {len(filtered_df)} records")
        else:
            df_plot = filtered_df
        
        if x_transform:
            try:
                df_plot['x_transformed'] = apply_transform(df_plot, x_axis, x_transform)
                x_col = 'x_transformed'
                x_label = f"{x_transform.replace('x', x_axis)}"
            except (ValueError, TypeError, KeyError, AttributeError):
                x_col = x_axis
                x_label = x_axis
                st.warning(f"X transform failed: {x_transform}")
        else:
            x_col = x_axis
            x_label = x_axis
        
        if y_transform and y_axis:
            try:
                df_plot['y_transformed'] = apply_transform(df_plot, y_axis, y_transform)
                y_col = 'y_transformed'
                y_label = f"{y_transform.replace('x', y_axis)}"
            except (ValueError, TypeError, KeyError, AttributeError):
                y_col = y_axis
                y_label = y_axis
                st.warning(f"Y transform failed: {y_transform}")
        else:
            y_col = y_axis
            y_label = y_axis
        
        # Use size_by directly without any transformation
        size_col = size_by
        
        # Create plot
        fig = None
        
        if plot_type == "scatter":
            # Handle trendlines if enabled
            if show_trendlines and trend_by:
                # Prepare trendline groups
                line_group_col = trend_by
                
                # If numeric column, bin it for grouping
                if pd.api.types.is_numeric_dtype(df_plot[trend_by]) and trend_bins:
                    df_plot['trend_group'] = bin_continuous_var(df_plot, trend_by, trend_bins)
                    line_group_col = 'trend_group'
                
                # Sort data by x-axis for proper line connections
                df_plot = df_plot.sort_values([line_group_col, x_col])
                
                # First add trendlines (so they appear behind points)
                fig = go.Figure()
                
                # Add trendlines by grouping data and adding line traces
                for group_name in df_plot[line_group_col].unique():
                    group_data = df_plot[df_plot[line_group_col] == group_name].sort_values(x_col)
                    
                    # Add line trace for this group
                    fig.add_trace(go.Scatter(
                        x=group_data[x_col],
                        y=group_data[y_col],
                        mode='lines',
                        name=f'Trend: {group_name}',
                        line=dict(width=2),
                        showlegend=True,
                        opacity=line_opacity
                    ))
                
                # Conditionally add scatter points on top (if enabled)
                if show_points_with_lines:
                    scatter_fig = px.scatter(
                        df_plot, x=x_col, y=y_col,
                        color=color_by, 
                        size=size_col,
                        size_max=max_size if size_col else None,
                        opacity=opacity,
                        title=f"{clean_label(y_label) if y_label else 'Data'} vs {clean_label(x_label)} (with trendlines)"
                    )
                    
                    # Apply uniform size when no size column is selected
                    if not size_col:
                        scatter_fig.update_traces(marker=dict(size=max_size))
                    
                    # Add scatter traces to the figure (points on top)
                    for trace in scatter_fig.data:
                        # Make points more prominent with border
                        trace.update(
                            marker=dict(
                                line=dict(width=1, color='white'),  # White border around points
                                opacity=opacity
                            )
                        )
                        fig.add_trace(trace)
                    
                    # Update layout from scatter figure
                    fig.update_layout(
                        title=scatter_fig.layout.title,
                        xaxis=scatter_fig.layout.xaxis,
                        yaxis=scatter_fig.layout.yaxis,
                        # Position legend to avoid overlap with colorbar
                        legend=dict(
                            x=1.02,
                            y=1,
                            xanchor='left',
                            yanchor='top'
                        )
                    )
                    
                    # If there's a colorbar from continuous color, position it properly
                    if color_by and pd.api.types.is_numeric_dtype(df_plot[color_by]):
                        fig.update_layout(
                            coloraxis_colorbar=dict(
                                x=1.02,
                                y=0.5,
                                xanchor='left',
                                yanchor='middle',
                                len=0.5
                            )
                        )
                else:
                    # Just lines, no points - set basic title and axes
                    fig.update_layout(
                        title=f"{clean_label(y_label) if y_label else 'Data'} vs {clean_label(x_label)} (trendlines only)",
                        xaxis_title=clean_label(x_label),
                        yaxis_title=clean_label(y_label) if y_label else ""
                    )
            else:
                # Regular scatter plot
                fig = px.scatter(
                    df_plot, x=x_col, y=y_col,
                    color=color_by, 
                    size=size_col,
                    size_max=max_size if size_col else None,
                    opacity=opacity,
                    title=f"{clean_label(y_label) if y_label else 'Data'} vs {clean_label(x_label)}"
                )
                
                # Apply uniform size when no size column is selected
                if not size_col:
                    fig.update_traces(marker=dict(size=max_size))
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
            fig.update_traces(boxmean=True)
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
            
            st.plotly_chart(fig, width='stretch')
        
        # Statistics
        st.subheader("Statistics")
        
        # Use the appropriate dataframe for statistics
        stats_df = df_plot
        
        stats_html = f"<b>Records displayed:</b> {len(stats_df)}<br>"
        if group_by_col:
            stats_html += f"<b>Groups:</b> {len(stats_df)} (from {len(filtered_df)} original records)<br>"
            stats_html += f"<b>Aggregation:</b> {agg_method}<br>"
        
        if x_axis in stats_df.columns and pd.api.types.is_numeric_dtype(stats_df[x_axis]):
            x_min, x_max = stats_df[x_axis].min(), stats_df[x_axis].max()
            stats_html += f"<b>X range:</b> [{x_min:.3g}, {x_max:.3g}]<br>"
        if y_axis and y_axis in stats_df.columns and pd.api.types.is_numeric_dtype(stats_df[y_axis]):
            y_min, y_max, y_mean = stats_df[y_axis].min(), stats_df[y_axis].max(), stats_df[y_axis].mean()
            stats_html += f"<b>Y range:</b> [{y_min:.3g}, {y_max:.3g}]<br>"
            stats_html += f"<b>Y mean:</b> {y_mean:.4f}<br>"
        
        st.markdown(stats_html, unsafe_allow_html=True)
