"""Visualize the data with Streamlit and spaCy."""
import streamlit as st
from spacy import displacy
import srsly

FILES = ["drugs_training.jsonl", "drugs_eval.jsonl"]
LABEL = "DRUG"

HTML_WRAPPER = "<div style='border-bottom: 1px solid #ccc; padding: 20px 0'>{}</div>"
SETTINGS = {"style": "ent", "manual": True, "options": {"colors": {LABEL: "#d1bcff"}}}


@st.cache(allow_output_mutation=True)
def load_data(filepath):
    return list(srsly.read_jsonl(filepath))


st.sidebar.title("Data visualizer")
st.sidebar.markdown(
    "Visualize the annotations using [displaCy](https://spacy.io/usage/visualizers) "
    "and view stats about the datasets."
)
data_file = st.sidebar.selectbox("Dataset", FILES)
data = load_data(data_file)
n_no_ents = 0
n_total_ents = 0

st.header(f"{data_file} ({len(data)})")
for eg in data:
    row = {"text": eg["text"], "ents": eg.get("spans", [])}
    n_total_ents += len(row["ents"])
    if not row["ents"]:
        n_no_ents += 1
    html = displacy.render(row, **SETTINGS).replace("\n\n", "\n")
    st.markdown(HTML_WRAPPER.format(html), unsafe_allow_html=True)

st.sidebar.markdown(
    f"""
| `{data_file}` | |
| --- | ---: |
| Total examples | {len(data):,} |
| Total entities | {n_total_ents:,} |
| Examples with no entities | {n_no_ents:,} |
"""
)
