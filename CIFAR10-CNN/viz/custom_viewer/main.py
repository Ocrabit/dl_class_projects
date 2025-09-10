import streamlit as st
from drop_in import drop_in_page
from base_viewer import create_base_viewer

def main():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'upload'
    
    if st.session_state.current_page == 'upload':
        existing_df = st.session_state.get('stored_df', None)
        result = drop_in_page(existing_df)
        
        if result is not None:
            st.session_state.stored_df = result
            st.session_state.selected_columns = result.columns.tolist()
            st.session_state.current_page = 'analysis'
            st.rerun()
    
    elif st.session_state.current_page == 'analysis':
        create_base_viewer(st.session_state.stored_df, stage="main")

if __name__ == "__main__":
    main()
