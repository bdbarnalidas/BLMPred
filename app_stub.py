"""
app_stub.py
------------------
This is a placeholder script for the future BLMPred web application.
The final implementation will provide an interactive interface for
peptide-level B-cell epitope prediction using ProtT5 embeddings and the
trained SVM classifier.

Planned features:
- Text-box input for peptide sequences
- Batch upload of peptides (FASTA/CSV)
- Automatic ProtT5 embedding generation
- BLMPred prediction output with confidence scores
- Visualization of feature importance (top dimensions)

The full web interface will be developed using Streamlit or Flask
in the next phase of the project.
"""

# -------------------------------------------------------------
# Imports (Streamlit or Flask will be enabled in final release)
# -------------------------------------------------------------
try:
    import streamlit as st
except ImportError:
    st = None  # Streamlit not installed yet


def main():
    if st is None:
        print(
            "BLMPred Web App Stub\n"
            "---------------------\n"
            "Streamlit is not installed in this environment.\n"
            "This is a placeholder script. The full web interface\n"
            "will be implemented in a future version of BLMPred.\n"
        )
        return

    # Web interface placeholder
    st.title("BLMPred Web Application (Stub)")
    st.write(
        """
        This is a placeholder for the upcoming BLMPred web application.

        Features that will be added:
        - Upload peptide sequences
        - Generate ProtT5 embeddings
        - Predict B-cell epitopes using the trained SVM
        - Display confidence scores and visualizations
        """
    )

    st.info("The full interface will be released in the next version.")


if __name__ == "__main__":
    main()
