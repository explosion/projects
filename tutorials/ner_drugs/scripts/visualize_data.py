"""Visualize the data with Streamlit and spaCy."""
import streamlit as st
from spacy import displacy
import srsly
import typer


@st.cache(allow_output_mutation=True)
def load_data(filepath):
    examples = list(srsly.read_jsonl(filepath))
    rows = []
    n_total_ents = 0
    n_no_ents = 0
    labels = set()
    for eg in examples:
        row = {"text": eg["text"], "ents": eg.get("spans", [])}
        n_total_ents += len(row["ents"])
        if not row["ents"]:
            n_no_ents += 1
        labels.update([span["label"] for span in row["ents"]])
        rows.append(row)
    return rows, labels, n_total_ents, n_no_ents


def main(file_paths: str):
    files = [p.strip() for p in file_paths.split(",")]
    st.sidebar.title("Data visualizer")
    st.sidebar.markdown(
        "Visualize the annotations using [displaCy](https://spacy.io/usage/visualizers) "
        "and view stats about the datasets."
    )
    data_file = st.sidebar.selectbox("Dataset", files)
    data, labels, n_total_ents, n_no_ents = load_data(data_file)
    displacy_settings = {
        "style": "ent",
        "manual": True,
        "options": {"colors": {label: "#d1bcff" for label in labels}},
    }
    st.header(f"{data_file} ({len(data)})")
    wrapper = "<div style='border-bottom: 1px solid #ccc; padding: 20px 0'>{}</div>"
    for row in data:
        html = displacy.render(row, **displacy_settings).replace("\n\n", "\n")
        st.markdown(wrapper.format(html), unsafe_allow_html=True)

    st.sidebar.markdown(
        f"""
    | `{data_file}` | |
    | --- | ---: |
    | Total examples | {len(data):,} |
    | Total entities | {n_total_ents:,} |
    | Examples with no entities | {n_no_ents:,} |
    """
    )


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass
