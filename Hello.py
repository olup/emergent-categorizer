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
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List
import instructor
from pydantic import BaseModel

LOGGER = get_logger(__name__)


load_dotenv()

# Initialize the OpenAI client from .env
openaiInstructorClient = instructor.from_openai(
    OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
)
openaiClient = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Category(BaseModel):
    label: str


# Define a function to get embeddings for a list of questions, by batch of 20
def get_question_embeddings(questions: list[str]):
    embeddings: List[List[float]] = []
    batch_size = 20
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i: i + batch_size]
        print(batch_questions)
        response = openaiClient.embeddings.create(
            model="text-embedding-3-small", input=batch_questions
        )
        embeddings.extend(map(lambda x: x.embedding, response.data))

    return embeddings


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Emergent categorizer")
    st.write("You don't know how to group your questions? Let the AI do it for you!")

    csv_file = st.file_uploader(
        "Qurestion list (CSV - one column only)", type=["csv"])
    if csv_file is None:
        return

    input_df = pd.read_csv(csv_file)

    content_column = st.selectbox("Content collumn", options=input_df.columns)
    if content_column is None:
        return

    df = pd.DataFrame()
    df["question"] = input_df[content_column]
    df = df.dropna()

    st.table(df["question"][:10])

    st.write(f"{len(df['question'])} questions loaded.")

    dataset_description = st.text_input(
        "Dataset description",
        placeholder="In a few word, describe the context of this dataset",
        help="In a few word, describe the context of this dataset",
    )

    category_count: int = st.number_input(
        "Number of categories",
        value=3,
        help="How many categories to infer from the list",
    ).__floor__()

    sample_size: int = st.number_input(
        "Number of examples to infer label",
        value=6,
        help="How many examples will be given to the LLM to infer the label of the category",
    ).__floor__()

    output_language = st.text_input(
        "Output language", "FRENCH", help="The language of the output labels"
    )

    print(output_language)
    print(sample_size)
    print(category_count)
    print(df["question"][:10])

    run = st.button("Run")
    if run == False:
        return

    st.write("Embedding questions...")
    question_embeddings = get_question_embeddings(df["question"].tolist())
    print(question_embeddings)

    df["embedding"] = question_embeddings
    emb_matrix = np.vstack(question_embeddings)

    st.write("Clustering...")
    kmeans = KMeans(n_clusters=category_count, random_state=42, init="k-means++").fit(
        emb_matrix
    )
    labels = kmeans.labels_
    df["category"] = labels

    st.write("Labeling...")

    # build a single prompt
    prompt = f"""
    We are interested in a specific dataset - here is the dataset's description:
    ---
    {dataset_description}
    ---

    The dataset was divided into categories.
    You are given a list of questions, randomly sampled from a category.
    You need to come up with a label for the category.
    Labels should be in : {output_language}.
    Here is the sample:
    ---
    """

    for i in range(category_count):
        if df[df["category"] == i].shape[0] < sample_size:
            sample_size = df[df["category"] == i].shape[0]
        category_questions_sample = (
            df[df["category"] == i]["question"]
            .sample(sample_size, random_state=42)
            .tolist()
        )
        prompt += "".join(f"{row}" + "\n" for row in category_questions_sample)
        prompt += """"
        ---
        """

    response = openaiInstructorClient.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a specialist analyst."},
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_model=List[Category],
    )

    st.write("Task complete")

    for i, category in enumerate(response):
        df.loc[df["category"] == i, "category_label"] = category.label

    # show a table with one line per category and the total number of questions in each category
    st.write(df.groupby("category_label").agg({"question": "count"}))

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
