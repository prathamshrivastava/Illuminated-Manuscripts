# app.py
import streamlit as st
import pandas as pd
from main import EnhancedTextSummarizer

def main():
    st.set_page_config(page_title="AI Text Summarizer", layout="wide")
    st.title("📑 Enhanced Text Summarizer")

    summarizer = EnhancedTextSummarizer()

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF or Text File", type=["pdf", "txt"])

    # Number of sentences
    num_sentences = st.slider("Number of sentences in summary", min_value=1, max_value=10, value=3)

    if uploaded_file is not None:
        try:
            # Extract text depending on file type
            if uploaded_file.type == "application/pdf":
                text = summarizer.extract_text_from_pdf(uploaded_file)
            else:
                text = uploaded_file.read().decode("utf-8")

            st.subheader("📄 Original Text Statistics")
            stats = summarizer.get_text_statistics(text)
            st.json(stats)

            # Generate summaries
            summaries = summarizer.get_all_summaries(text, num_sentences)

            # Show summaries
            st.subheader("📝 Generated Summaries")
            for technique, summary in summaries.items():
                with st.expander(f"{technique}"):
                    st.write(summary)

            # Evaluate summaries
            st.subheader("📊 Evaluation Metrics")
            evaluation_df = summarizer.evaluate_summaries(summaries, text)
            st.dataframe(evaluation_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        st.info("👆 Please upload a file to start summarizing.")

if __name__ == "__main__":
    main()
