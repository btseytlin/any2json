import streamlit as st
import json


def get_truncated_data(data, length: int = 50) -> str:
    if isinstance(data, str):
        return f"{data[:length]}..." if len(data) > length else data
    elif isinstance(data, (dict, list)):
        json_str = json.dumps(data)
        return f"{json_str[:length]}..." if len(json_str) > length else json_str
    return f"{str(data)[:length]}..." if len(str(data)) > length else str(data)


def main():
    st.set_page_config(layout="wide")
    st.title("samples.json viewer")

    with open("data/intermediate/samples.json", "r") as f:
        samples = json.load(f)

    if not samples:
        st.warning("samples.json is empty.")
        return

    table_data = []
    for i, sample in enumerate(samples):
        table_data.append(
            {
                **sample,
            }
        )

    st.subheader("All Samples Overview")
    st.dataframe(
        table_data,
        use_container_width=True,
        column_config={
            "input_data": st.column_config.TextColumn(width="medium"),
            "output": st.column_config.TextColumn(width="medium"),
            "schema": st.column_config.TextColumn(width="medium"),
        },
        hide_index=False,
    )


if __name__ == "__main__":
    main()
