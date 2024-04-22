# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Initialize the OpenAI client from .env
openaiClient = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))


# Define a function to get embeddings for a list of questions, by batch of 20
def get_question_embeddings(questions):
    embeddings: List[List[float]] = []
    batch_size = 20
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i : i + batch_size]
        response = openaiClient.embeddings.create(
            model="text-embedding-3-small", input=batch_questions
        )
        embeddings.extend(map(lambda x: x.embedding, response.data))

    return embeddings


LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Emergent ctaegorizer")
    st.write("This app lets you categorize questions into auto identified categories.")

    csv_file = st.file_uploader("Qurestion list (CSV - one column only)", type=["csv"])
    if csv_file is None:
        return

    csv_content = csv_file.getvalue().decode("utf-8")
    lines = csv_content.split("\n")

    headers = lines[0].split(",")

    lines = lines[1:]
    st.write("Number of questions: ", len(lines))
    content_column = st.selectbox("Content collumn", headers)
    if content_column is None:
        return

    content_column_index = headers.index(content_column)

    questions = []
    for line in lines:
        if len(line) == 0:
            continue
        questions.append(line.split(",")[content_column_index])

    df = pd.DataFrame(questions, columns=["question"])

    st.table(df["question"][:10])

    dataset_description = st.text_input(
        "Dataset description", "In a few word, describe the context of this dataset"
    )

    number: int = st.number_input(
        "Number of categories",
        value=3,
        placeholder="How many categories to infer from the list",
    ).__floor__()

    run = st.button("Run")
    if run == False:
        return

    st.write("Embedding questions...")
    question_embeddings = get_question_embeddings(questions)
    df["embedding"] = question_embeddings

    emb_matrix = np.vstack(question_embeddings)
    st.write(emb_matrix.shape)

    st.write("Clustering...")
    kmeans = KMeans(n_clusters=number, random_state=42, init="k-means++").fit(
        emb_matrix
    )
    labels = kmeans.labels_

    df["category"] = labels

    st.write("Done!")

    # ask chatgpt to label the categories
    for i in range(number):
        sample_size = 5
        if df[df["category"] == i].shape[0] < sample_size:
            sample_size = df[df["category"] == i].shape[0]
        category_questions_sample = (
            df[df["category"] == i]["question"]
            .sample(sample_size, random_state=42)
            .tolist()
        )
        response = openaiClient.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a specialist analyst."},
                {
                    "role": "user",
                    "content": """
                    We are interested in a specific dataset - here is the dataset's description:
                    ---
                    """
                    + dataset_description
                    + """
                    ---

                    The dataset was divided into categories.
                    You are given a list of questions, randomly sampled from a category.
                    You need to come up with a label for the category. You speak the language used in the dataset description.
                    Here is the sample:
                    ---
                    """
                    + "".join(f"{row}" + "\n" for row in category_questions_sample)
                    + """"
                    ---
                    
                    You will output ONLY a json object with \{ "category": "label" \} with no code fence.
                    """,
                },
            ],
        )
        category_label = response.choices[0].message.content
        st.write(f"{category_label}")
        df.loc[df["category"] == i, "category_label"] = category_label

    # download csv file
    st.download_button(
        "Press to Download",
        df[["question", "category"]].to_csv().encode("utf-8"),
        "file.csv",
        "text/csv",
        key="download-csv",
    )


if __name__ == "__main__":
    run()
