import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Default maximum point size for scatter plots
DEFAULT_MAX_SIZE = 5

# Maximum number of unique values for hyperparameter filtering
MAX_HYPERPARAM_UNIQUE_VALUES = 15

# Threshold for treating numeric columns as discrete (use legend instead of colorbar)
DISCRETE_NUMERIC_THRESHOLD = 10

def clean_label(label):
    """Clean and format labels for plots"""
    return label.replace("_", " ").title()

def prepare_color_column(df_plot, color_by):
    """Prepare color column for plotting, converting discrete numerics to strings"""
    if not color_by:
        return color_by
    
    # Check if numeric column should be treated as discrete/categorical
    if (pd.api.types.is_numeric_dtype(df_plot[color_by]) and 
        df_plot[color_by].nunique() <= DISCRETE_NUMERIC_THRESHOLD):
        # Convert to string for discrete treatment
        df_plot[f'{color_by}_str'] = df_plot[color_by].astype(str)
        return f'{color_by}_str'
    
    return color_by


def validate_transform(df, col, transform_str):
    """Validate a transformation without applying it to full data"""
    if not transform_str or not transform_str.strip():
        return True, ""
    
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False, "Column must be numeric for transforms"
    
    try:
        sample = df[col].head(10).copy()
        # Use both x and y as variable names so transforms work for both axes
        x = sample
        y = sample
        allowed_names = {
            "x": x, "y": y, "np": np, "pd": pd,
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
        
        return True, "✓ Valid"
    except Exception as e:
        return False, str(e)[:40]

def apply_transform(df, col, transform_str):
    """Apply mathematical transformation to a column"""
    if not transform_str or not transform_str.strip():
        return df[col].copy()
    
    try:
        # Use both x and y as variable names so transforms work for both axes
        x = df[col].copy()
        y = df[col].copy()
        allowed_names = {
            "x": x, "y": y, "np": np, "pd": pd,
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

def build_advanced_filter(cat_filter_col, selected_vals, hyperparam_selections):
    """Build filter expression from advanced filter selections"""
    all_conditions = []
    
    # Add categorical filter conditions
    if cat_filter_col and selected_vals:
        for val in selected_vals:
            all_conditions.append(build_condition(cat_filter_col, val))
    
    # Add hyperparameter filter conditions (each selection as independent OR group)
    for selection in hyperparam_selections:
        if selection:
            # Build AND condition for this selection
            selection_conditions = [build_condition(col, val) for col, val in selection.items()]
            if selection_conditions:
                # Each selection is an AND group, but selections are OR'd together
                all_conditions.append(f"({' & '.join(selection_conditions)})")
    
    # Join all conditions with OR
    return f"({' | '.join(all_conditions)})" if all_conditions else ""

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

def build_condition(col, val):
    """Build a single filter condition for a column and value"""
    if isinstance(val, str):
        escaped_val = val.replace("'", "\\'")
        return f"`{col}` == '{escaped_val}'"
    elif pd.isna(val):
        return f"`{col}`.isna()"
    else:
        return f"`{col}` == {val}"

def create_scatter_plot(df_plot, x_col, y_col, color_column, size_col, max_size, opacity, hover_fields, title):
    """Create a scatter plot with consistent formatting"""
    fig = px.scatter(
        df_plot, x=x_col, y=y_col,
        color=color_column, 
        size=size_col,
        size_max=max_size if size_col else None,
        opacity=opacity,
        hover_data=hover_fields if hover_fields else None,
        title=title
    )
    
    if not size_col:
        fig.update_traces(marker=dict(size=max_size))
    
    return fig

def prepare_x_for_categorical_plot(df_plot, x_col, n_bins):
    """Prepare X axis for categorical plots by binning if numeric"""
    if pd.api.types.is_numeric_dtype(df_plot[x_col]):
        df_plot['x_binned'] = bin_continuous_var(df_plot, x_col, n_bins)
        return 'x_binned'
    return x_col

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

def select_fun_initial_columns(df, num_cols, cat_cols):
    """Select fun initial cols for x and y axes
    
    Returns:
        tuple: (default_x, default_y) column names
    """
    # For X-axis: prefer few discrete values
    default_x = None
    
    # First choice: categorical columns
    choose_cat_or_disc = np.random.choice([0, 1, 1, 1, 1])  # Meh I like non cat more
    if choose_cat_or_disc:
        # print("Choosing a categorical column for X")
        index_greater_than_3_cols = [
            i
            for i, col in enumerate(cat_cols)
            if (
                    df[col].nunique() > 3
                    and (
                            not pd.api.types.is_string_dtype(df[col])
                            or df[col].str.len().max() < 20
                    )
            )
        ]

        if not index_greater_than_3_cols:
            index_greater_than_3_cols = [0]
        
        default_x = cat_cols[np.random.choice(index_greater_than_3_cols)]
    else:  # Okay discrete we go
        # print("Choosing a discrete column for X")
        if num_cols:
            num_cols_filtered = [col for col in num_cols if df[col].nunique(dropna=True) > 3]

            if num_cols_filtered:
                default_x = np.random.choice(num_cols_filtered)
            else:
                default_x = num_cols[0]
        else:
            default_x = df.columns[0] if len(df.columns) > 0 else None
    
    # For Y-axis: prefer continuous variables (high cardinality numeric)
    default_y = num_cols[0] if num_cols else (cat_cols[0] if cat_cols else None)

    if num_cols:
        # Sort numeric columns by number of unique values (descending for high cardinality)
        num_cols_by_cardinality = sorted(
            num_cols,
            key=lambda col: df[col].nunique(dropna=True),
            reverse=True
        )

        # Pick randomly from the top-k highest cardinality numeric columns
        k = min(3, len(num_cols_by_cardinality))  # top 3 or fewer
        candidates = num_cols_by_cardinality[:k]

        # If x already taken, filter it out (unless it’s the only option)
        candidates = [col for col in candidates if col != default_x] or candidates

        default_y = np.random.choice(candidates)
    
    return default_x, default_y

def create_base_viewer(df, stage="base", use_fun_defaults=False):
    """Create a generic data visualization dashboard
    
    Args:
        df: DataFrame to visualize
        stage: Unique identifier for the viewer instance
        use_fun_defaults: Whether to use random fun initial column selection
    """
    if df is None or df.empty:
        st.warning("No data available")
        return
    
    col_back, col_reset, col_spacer, col_fun = st.columns([2, 2, 10, 1])
    with col_back:
        if st.button("← Back to Data Upload", type="secondary", key=f"{stage}_back"):
            st.session_state.current_page = 'upload'
            if 'selected_columns' in st.session_state:
                del st.session_state.selected_columns
            st.rerun()
    
    with col_reset:
        if st.button("Reset Fields", type="secondary", key=f"{stage}_reset"):
            viewer_key = f"{stage}_viewer_selections"
            if viewer_key in st.session_state:
                del st.session_state[viewer_key]
            keys_to_clear = [key for key in st.session_state.keys() if key.startswith(f"{stage}_")]
            for key in keys_to_clear:
                del st.session_state[key]
            st.rerun()

    with col_fun:
        if st.button("🎲!", type="secondary", key=f"{stage}_fun", width='stretch'):
            viewer_key = f"{stage}_viewer_selections"
            if viewer_key in st.session_state:
                del st.session_state[viewer_key]
            st.session_state[f"{stage}_roll_fun"] = True
            st.rerun()

    # Auto-detect column types (cache these as they're used multiple times)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = cat_cols + num_cols
    
    # Initialize viewer selections in session state if not present
    viewer_key = f"{stage}_viewer_selections"
    
    # Check for fun roll
    roll_fun_key = f"{stage}_roll_fun"
    should_use_fun = use_fun_defaults or st.session_state.get(roll_fun_key, False)
    
    # Clean it
    if roll_fun_key in st.session_state:
        del st.session_state[roll_fun_key]
    
    if viewer_key not in st.session_state:
        if should_use_fun:
            default_x, default_y = select_fun_initial_columns(df, num_cols, cat_cols)
        else:
            default_x = num_cols[0] if num_cols else cat_cols[0] if cat_cols else df.columns[0]
            default_y = num_cols[1] if len(num_cols) > 1 else num_cols[0] if num_cols else None
        default_color = cat_cols[0] if cat_cols else None
        
        st.session_state[viewer_key] = {
            'x_axis': default_x,
            'y_axis': default_y,
            'color_by': default_color,
            'size_by': None,
            'plot_type': 'scatter',
            'opacity': 0.7,
            'n_bins': 10,
            'group_by': None,
            'agg_method': 'mean',
            'filter_text': '',
            'x_transform': '',
            'y_transform': '',
            'filter_mode': 'Categorical Values',
            'cat_filter_col': None,
            'cat_filter_vals': [],
            'show_trendlines': False,
            'trend_by': None,
            'trend_bins': 5,
            'line_opacity': 0.85,
            'show_points_with_lines': True,
            'hover_fields': [],
            'size_scale': DEFAULT_MAX_SIZE,
            'hide_single_values': True,
            'hyper_selections': []
        }
    
    # Get stored selections
    selections = st.session_state[viewer_key]
    
    # Function to update stored selections when widgets change
    def update_selections():
        # Only update stored values FROM widget session state (one-way sync)
        if f"{stage}_plot_type" in st.session_state:
            selections['plot_type'] = st.session_state[f"{stage}_plot_type"]
        if f"{stage}_x_axis" in st.session_state:
            selections['x_axis'] = st.session_state[f"{stage}_x_axis"]
        if f"{stage}_y_axis" in st.session_state:
            selections['y_axis'] = st.session_state[f"{stage}_y_axis"]
        if f"{stage}_color" in st.session_state:
            selections['color_by'] = st.session_state[f"{stage}_color"]
        if f"{stage}_size" in st.session_state:
            selections['size_by'] = st.session_state[f"{stage}_size"]
        if f"{stage}_opacity" in st.session_state:
            selections['opacity'] = st.session_state[f"{stage}_opacity"]
        if f"{stage}_bins" in st.session_state:
            selections['n_bins'] = st.session_state[f"{stage}_bins"]
        if f"{stage}_group_by" in st.session_state:
            selections['group_by'] = st.session_state[f"{stage}_group_by"]
        if f"{stage}_agg_method" in st.session_state:
            selections['agg_method'] = st.session_state[f"{stage}_agg_method"]
        if f"{stage}_filter" in st.session_state:
            selections['filter_text'] = st.session_state[f"{stage}_filter"]
        if f"{stage}_x_transform" in st.session_state:
            selections['x_transform'] = st.session_state[f"{stage}_x_transform"]
        if f"{stage}_y_transform" in st.session_state:
            selections['y_transform'] = st.session_state[f"{stage}_y_transform"]
        if f"{stage}_filter_mode" in st.session_state:
            selections['filter_mode'] = st.session_state[f"{stage}_filter_mode"]
        if f"{stage}_cat_filter_col" in st.session_state:
            selections['cat_filter_col'] = st.session_state[f"{stage}_cat_filter_col"]
        if f"{stage}_cat_filter_vals" in st.session_state:
            selections['cat_filter_vals'] = st.session_state[f"{stage}_cat_filter_vals"]
        if f"{stage}_show_trendlines" in st.session_state:
            selections['show_trendlines'] = st.session_state[f"{stage}_show_trendlines"]
        if f"{stage}_trend_by" in st.session_state:
            selections['trend_by'] = st.session_state[f"{stage}_trend_by"]
        if f"{stage}_trend_bins" in st.session_state:
            selections['trend_bins'] = st.session_state[f"{stage}_trend_bins"]
        if f"{stage}_line_opacity" in st.session_state:
            selections['line_opacity'] = st.session_state[f"{stage}_line_opacity"]
        if f"{stage}_show_points_with_lines" in st.session_state:
            selections['show_points_with_lines'] = st.session_state[f"{stage}_show_points_with_lines"]
        if f"{stage}_hover_fields" in st.session_state:
            selections['hover_fields'] = st.session_state[f"{stage}_hover_fields"]
        if f"{stage}_size_scale" in st.session_state:
            selections['size_scale'] = st.session_state[f"{stage}_size_scale"]
        if f"{stage}_hide_single_values" in st.session_state:
            selections['hide_single_values'] = st.session_state[f"{stage}_hide_single_values"]
        if f"{stage}_hyper_selections" in st.session_state:
            selections['hyper_selections'] = st.session_state[f"{stage}_hyper_selections"]
    
    update_selections()
    col1, col2 = st.columns([1, 4], gap="large")
    
    with col1:
        with st.container(height="stretch"):
            st.subheader("Controls")
            
            plot_options = ["scatter", "histogram", "boxplot", "violin", "hexbin"]
            
            # Plot type
            plot_type = st.selectbox(
                "Plot Type",
                plot_options,
                index=plot_options.index(selections.get('plot_type', 'scatter')),
                key=f"{stage}_plot_type"
            )

            # Axis selectors
            x_axis = st.selectbox(
                "X Axis",
                df.columns.tolist(),
                index=df.columns.tolist().index(selections['x_axis']) if selections['x_axis'] in df.columns else 0,
                key=f"{stage}_x_axis"
            )

            if num_cols:
                y_axis = st.selectbox(
                    "Y Axis",
                    num_cols,
                    index=num_cols.index(selections['y_axis']) if selections['y_axis'] in num_cols else 0,
                    key=f"{stage}_y_axis"
                )
            else:
                y_axis = None
                st.warning("No numeric columns found for Y-axis")

            # Transform inputs
            x_transform = st.text_input(
                "X Transform",
                value=selections.get('x_transform', ''),
                placeholder="np.log10(x) or log(x)",
                key=f"{stage}_x_transform"
            )
            y_transform = st.text_input(
                "Y Transform",
                value=selections.get('y_transform', ''),
                placeholder="y*100 or log(y)",
                key=f"{stage}_y_transform"
            )

            # Validate transforms and clear malformed ones from session state
            if x_transform and x_axis:
                is_valid, message = validate_transform(df, x_axis, x_transform)
                if is_valid:
                    st.success(f"X: {message}")
                else:
                    st.error(f"X: {message}")
                    # Clear malformed transform from session state
                    if f"{stage}_x_transform" in st.session_state:
                        del st.session_state[f"{stage}_x_transform"]
            
            if y_transform and y_axis:
                is_valid, message = validate_transform(df, y_axis, y_transform)
                if is_valid:
                    st.success(f"Y: {message}")
                else:
                    st.error(f"Y: {message}")
                    # Clear malformed transform from session state
                    if f"{stage}_y_transform" in st.session_state:
                        del st.session_state[f"{stage}_y_transform"]

            # Encoding options in expander

            with st.expander("Visual Specifications", expanded=True):
                if plot_type != "hexbin":
                    color_by = st.selectbox(
                        "Color By",
                        [None] + all_cols,
                        index=([None] + all_cols).index(selections['color_by']) if selections['color_by'] in all_cols else 0,
                        key=f"{stage}_color"
                    )
                else:
                    color_by = None

                if plot_type == "scatter":
                    size_by = st.selectbox(
                        "Size By",
                        [None] + num_cols,
                        index=([None] + num_cols).index(selections['size_by']) if selections['size_by'] in ([None] + num_cols) else 0,
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
                        min_value=0.1, max_value=1.0, value=selections.get('opacity', 0.7), step=0.1,
                        key=f"{stage}_opacity"
                    )
                else:
                    opacity = 0.7

                # Hover info selector for all plot types
                hover_fields = st.multiselect(
                    "Hover Fields",
                    df.columns.tolist(),
                    default=selections.get('hover_fields', []),
                    key=f"{stage}_hover_fields",
                    help="Additional columns to show when hovering over data points"
                )

            # Trendlines feature in expandable section (only for scatter plots)
            if plot_type == "scatter":
                with st.expander("Trendlines", expanded=False):
                    show_trendlines = st.checkbox(
                        "Show Trendlines",
                        value=selections.get('show_trendlines', False),
                        key=f"{stage}_show_trendlines",
                        help="Connect points into lines based on grouping"
                    )

                    if show_trendlines:
                        trend_by = st.selectbox(
                            "Line Groups",
                            df.columns.tolist(),
                            index=df.columns.tolist().index(selections['trend_by']) if selections.get('trend_by') in df.columns.tolist() else 0,
                            key=f"{stage}_trend_by",
                            help="Connect points with same value in this column"
                        )

                        # Check if trend column is numeric and needs binning
                        if trend_by and pd.api.types.is_numeric_dtype(df[trend_by]):
                            trend_bins = st.slider(
                                "Groups (bins)",
                                min_value=2,
                                max_value=20,
                                value=selections.get('trend_bins', 5),
                                key=f"{stage}_trend_bins",
                                help="Number of groups to divide continuous data into"
                            )
                        else:
                            trend_bins = None

                        line_opacity = st.slider(
                            "Line Opacity",
                            min_value=0.1,
                            max_value=1.0,
                            value=selections.get('line_opacity', 0.85),
                            step=0.1,
                            key=f"{stage}_line_opacity",
                            help="Opacity of the trendlines"
                        )

                        show_points_with_lines = st.checkbox(
                            "Show Points",
                            value=selections.get('show_points_with_lines', True),
                            key=f"{stage}_show_points_with_lines",
                            help="Show individual points in addition to trendlines"
                        )
            else:
                show_trendlines = False

            if plot_type in ["histogram", "hexbin"]:
                n_bins = st.slider(
                    "Bins",
                    min_value=5, max_value=20, value=selections.get('n_bins', 10), step=1,
                    key=f"{stage}_bins"
                )
            elif plot_type in ["boxplot", "violin"]:
                n_bins = st.slider(
                    "Bins (for continuous X)",
                    min_value=5, max_value=20, value=selections.get('n_bins', 10), step=1,
                    key=f"{stage}_bins",
                    help="Used to bin continuous X variables for categorical display"
                )

            # Filter
            st.subheader("Filter")
            filter_text = st.text_input(
                "Filter Expression",
                value=selections.get('filter_text', ''),
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
                filter_options = ["Categorical Values", "Hyperparameter Combinations"]
                filter_mode = st.radio(
                    "Filter Mode",
                    filter_options,
                    index=filter_options.index(selections.get('filter_mode', 'Categorical Values')),
                    key=f"{stage}_filter_mode",
                    help="Choose between categorical filtering or hyperparameter grid filtering"
                )

                if filter_mode == "Categorical Values":
                    st.markdown("**Select by Categorical Values**")

                    # Select categorical column to filter
                    cat_options = [None] + cat_cols + [col for col in df.columns if col not in num_cols and col not in cat_cols]
                    cat_filter_col = st.selectbox(
                        "Filter by Column",
                        cat_options,
                        index=cat_options.index(selections.get('cat_filter_col')) if selections.get('cat_filter_col') in cat_options else 0,
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
                            default=selections.get('cat_filter_vals', []),
                            key=f"{stage}_cat_filter_vals",
                            help="Select one or more values to include"
                        )

                else:  # Hyperparameter Combinations
                    st.markdown("**Hyperparameter Combinations**")
                    st.markdown("Filter by combinations of categorical and discrete values")
                    st.caption("Select values for one or more columns to create a filter combination")

                    # Checkbox to hide columns with only 1 value
                    hide_single_values = st.checkbox(
                        "Hide columns of 1",
                        value=selections.get('hide_single_values', True),
                        key=f"{stage}_hide_single_values",
                        help="Hide columns that have only one possible value based on current filters"
                    )

                    # Identify all discrete columns (both categorical and numeric with limited unique values)
                    discrete_cols = []

                    # Add categorical columns
                    for col in cat_cols:
                        if df[col].nunique() <= MAX_HYPERPARAM_UNIQUE_VALUES:
                            discrete_cols.append(col)

                    # Add discrete numeric columns
                    for col in num_cols:
                        if 1 < df[col].nunique() <= MAX_HYPERPARAM_UNIQUE_VALUES:
                            discrete_cols.append(col)

                    # Also include columns that aren't strictly numeric or categorical but have discrete values
                    for col in df.columns:
                        if col not in discrete_cols and df[col].nunique() <= MAX_HYPERPARAM_UNIQUE_VALUES:
                            discrete_cols.append(col)

                    if discrete_cols:
                        # Initialize selections list in session state if not exists
                        selections_key = f"{stage}_hyper_selections"
                        if selections_key not in st.session_state:
                            st.session_state[selections_key] = selections.get('hyper_selections', [])

                        current_selection = {}
                        for col in discrete_cols:
                            key = f"{stage}_hyper_new_{col}"
                            if key in st.session_state and st.session_state[key] not in [None, "Choose an option"]:
                                current_selection[col] = st.session_state[key]

                        # Apply current partial selection to get available data for dynamic filtering
                        if current_selection:
                            temp_conditions = [build_condition(col, val) for col, val in current_selection.items()]
                            if temp_conditions:
                                temp_filter = " & ".join(temp_conditions)
                                available_df = apply_filter(df, temp_filter)
                            else:
                                available_df = df
                        else:
                            available_df = df

                        # Show current selections
                        if st.session_state[selections_key]:
                            st.markdown("**Current Selections:**")
                            for i, selection in enumerate(st.session_state[selections_key]):
                                col_container = st.container()
                                with col_container:
                                    sel_cols = st.columns([3, 1])
                                    with sel_cols[0]:
                                        selection_text = " AND ".join([f"{k}={v}" for k, v in selection.items()])
                                        st.text(f"{i+1}. {selection_text}")
                                    with sel_cols[1]:
                                        if st.button("Remove", key=f"{stage}_remove_{i}"):
                                            st.session_state[selections_key].pop(i)
                                            st.rerun()

                        # Add new selection interface
                        new_selection = {}

                        # Dynamic Show of Hyperparameter Columns
                        for col in discrete_cols:
                            if col in available_df.columns:
                                available_vals = sorted(available_df[col].dropna().unique().tolist())

                                current_val = st.session_state.get(f"{stage}_hyper_new_{col}", "Choose an option")

                                if hide_single_values and len(available_vals) <= 1 and current_val == "Choose an option":
                                    continue

                                if available_vals:
                                    try:
                                        if current_val in available_vals:
                                            default_index = available_vals.index(current_val) + 1  # +1 for None at index 0
                                        else:
                                            default_index = 0  # Default to None
                                    except (ValueError, TypeError):
                                        default_index = 0

                                    # Show count only if nothing is selected
                                    if current_val == "Choose an option":
                                        label = f"{col} ({len(available_vals)} available)"
                                    else:
                                        label = col

                                    selected_val = st.selectbox(
                                        label,
                                        ["Choose an option"] + available_vals,
                                        index=default_index,
                                        key=f"{stage}_hyper_new_{col}",
                                        help=f"Select a value for {col} to include in this filter"
                                    )

                                    if selected_val != "Choose an option":
                                        new_selection[col] = selected_val

                        # Add selection button!
                        col_btns = st.columns(2)
                        with col_btns[0]:
                            if st.button("Add Selection", key=f"{stage}_add_selection",
                                        disabled=len(new_selection) == 0,
                                        help="Add current selection to filter"):
                                if new_selection:
                                    st.session_state[selections_key].append(new_selection)
                                    for col in discrete_cols:
                                        key = f"{stage}_hyper_new_{col}"
                                        if key in st.session_state:
                                            del st.session_state[key]
                                    st.rerun()

                        with col_btns[1]:
                            if st.button("Clear All", key=f"{stage}_clear_all"):
                                st.session_state[selections_key] = []
                                # Clear all input fields
                                for col in discrete_cols:
                                    key = f"{stage}_hyper_new_{col}"
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.rerun()

                        # Show statistics about filtering
                        if current_selection:
                            st.info(f"Available records after current filters: {len(available_df)} / {len(df)}")

                    else:
                        st.info(f"No discrete columns found for filtering (need ≤{MAX_HYPERPARAM_UNIQUE_VALUES} unique values)")

            # Group By
            st.subheader("Group By")
            group_by_col = st.selectbox(
                "Group By Column",
                [None] + df.columns.tolist(),
                index=([None] + df.columns.tolist()).index(selections.get('group_by')) if selections.get('group_by') in ([None] + df.columns.tolist()) else 0,
                key=f"{stage}_group_by",
                help="Group data by this column and aggregate"
            )

            if group_by_col:
                agg_options = ["mean", "median", "sum", "count", "min", "max"]
                agg_method = st.selectbox(
                    "Aggregation Method",
                    agg_options,
                    index=agg_options.index(selections.get('agg_method', 'mean')),
                    key=f"{stage}_agg_method",
                    help="How to aggregate numeric values within each group"
                )

    # Now start the viz section
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
            hyperparam_selections = []
        else:
            cat_filter_col = None
            selected_vals = []
            
            # Get hyperparameter selections directly from selections list
            selections_key = f"{stage}_hyper_selections"
            hyperparam_selections = st.session_state.get(selections_key, [])
        
        # Build advanced filter expression
        advanced_filter = build_advanced_filter(cat_filter_col, selected_vals, hyperparam_selections)
        
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
            except Exception as e:
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
            except Exception as e:
                y_col = y_axis
                y_label = y_axis
                st.warning(f"Y transform failed: {y_transform}")
        else:
            y_col = y_axis
            y_label = y_axis
        
        size_col = size_by
        
        color_column = prepare_color_column(df_plot, color_by)
        
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
                    
                    # Build hover info for trendlines
                    hover_template = f'<b>{clean_label(trend_by)}: {group_name}</b><br>'
                    hover_template += f'{clean_label(x_label)}: %{{x}}<br>'
                    hover_template += f'{clean_label(y_label) if y_label else "Y"}: %{{y}}<br>'
                    hover_template += f'<b>Points in trend:</b> {len(group_data)}<br>'
                    
                    # Add additional hover fields if selected
                    if not show_points_with_lines and hover_fields:
                        for field in hover_fields:
                            if field in group_data.columns:
                                field_val = group_data[field].dropna().iloc[0] if len(group_data[field].dropna()) > 0 else "N/A"
                                hover_template += f'{clean_label(field)}: {field_val}<br>'
                    
                    hover_template += '<extra></extra>'
                    
                    # Add line trace for this group
                    fig.add_trace(go.Scatter(
                        x=group_data[x_col],
                        y=group_data[y_col],
                        mode='lines',
                        name=str(group_name),
                        line=dict(width=2),
                        showlegend=True,
                        opacity=line_opacity,
                        hovertemplate=hover_template
                    ))
                
                # Conditionally add scatter points on top (if enabled)
                if show_points_with_lines:
                    scatter_fig = create_scatter_plot(
                        df_plot, x_col, y_col, color_column, size_col, max_size, 
                        opacity, hover_fields, 
                        f"{clean_label(y_label) if y_label else 'Data'} vs {clean_label(x_label)} (with trendlines)"
                    )

                    # TODO: Make a setting field to edit marker outline value
                    for trace in scatter_fig.data:
                        trace.update(
                            marker=dict(
                                line=dict(width=1, color='white'),
                                opacity=opacity
                            )
                        )
                        fig.add_trace(trace)

                    fig.update_layout(
                        title=scatter_fig.layout.title,
                        xaxis=scatter_fig.layout.xaxis,
                        yaxis=scatter_fig.layout.yaxis,
                        legend=dict(
                            tracegroupgap=15
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
                fig = create_scatter_plot(
                    df_plot, x_col, y_col, color_column, size_col, max_size, 
                    opacity, hover_fields, 
                    f"{clean_label(y_label) if y_label else 'Data'} vs {clean_label(x_label)}"
                )
        elif plot_type == "histogram":
            target_col = y_col if y_col else x_col
            
            fig = px.histogram(
                df_plot, x=target_col, color=color_column,
                nbins=n_bins,
                opacity=opacity,
                hover_data=hover_fields if hover_fields else None,
                title=f"Distribution of {clean_label(target_col)}"
            )
        elif plot_type == "boxplot":
            x_col_for_plot = prepare_x_for_categorical_plot(df_plot, x_col, n_bins)
            
            fig = px.box(
                df_plot, x=x_col_for_plot, y=y_col, color=color_column,
                hover_data=hover_fields if hover_fields else None,
                title=f"{clean_label(y_label)} by {clean_label(x_label)}"
            )
            fig.update_traces(boxmean=True)
        elif plot_type == "violin":
            x_col_for_plot = prepare_x_for_categorical_plot(df_plot, x_col, n_bins)
            
            fig = px.violin(
                df_plot, x=x_col_for_plot, y=y_col, color=color_column,
                hover_data=hover_fields if hover_fields else None,
                title=f"{clean_label(y_label)} by {clean_label(x_label)}"
            )
        elif plot_type == "hexbin":
            fig = go.Figure(data=go.Histogram2d(
                x=df_plot[x_col], y=df_plot[y_col],
                colorscale='Blues'
            ))
            fig.update_layout(title=f"{clean_label(y_label)} vs {clean_label(x_label)} (density)")
        
        if fig:
            # Centralized legend grouping for all plot types
            for trace in fig.data:
                # Group custom trendline traces (they have mode='lines' and are from trendlines)
                if show_trendlines and hasattr(trace, 'mode') and trace.mode == 'lines':
                    trace.update(
                        legendgroup="trends",
                        legendgrouptitle_text="Trendlines"
                    )
                # Group all other colored traces (scatter points, histograms, etc.)
                elif color_by:
                    trace.update(
                        legendgroup="colors",
                        legendgrouptitle_text=f"Color: {clean_label(color_by)}"
                    )
            
            # Clean up labels and set legend spacing
            fig.update_layout(
                xaxis_title=clean_label(x_label),
                yaxis_title=clean_label(y_label) if y_label else "",
                height=800,
                legend=dict(
                    tracegroupgap=15,
                    title=None
                ) if (show_trendlines or color_by) else {},
                coloraxis_colorbar=dict(x=1.1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
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
        
        # Add GitHub link at the bottom
        st.markdown("---")
        st.markdown("[View project source code on GitHub](https://github.com/Ocrabit/dl_class_projects/tree/main/CIFAR10-CNN)")
