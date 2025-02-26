import pandas as pd
import os
import logging
import re
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set upload folder for CSV files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store CSV content
csv_data = None


class Tools:
    def format_output(text):
        """Convert Markdown bold syntax to HTML <strong> tags."""
        return re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)


def initialize_llama3():
    try:
        create_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI assistant that provides structured, concise answers strictly based on the uploaded CSV file. "
                    "Format responses in a table where possible, and avoid unnecessary content.",
                ),
                ("user", "Question: {question}\nData: {data}"),
            ]
        )

        llama_model = Ollama(model="llama3.1:latest", temperature=0.2)
        output_parser = StrOutputParser()

        return create_prompt | llama_model | output_parser
    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {e}")
        return None


chatbot_pipeline = initialize_llama3()


def extract_relevant_data(query, df):
    """Extract relevant rows based on keywords in the query (up to 500 rows)."""
    query_keywords = query.lower().split()
    relevant_rows = df[
        df.apply(
            lambda row: any(keyword in str(row).lower() for keyword in query_keywords),
            axis=1,
        )
    ]

    if relevant_rows.empty:
        return "No relevant data found in the CSV."

    return relevant_rows.head(500).to_string(index=False)


def main():
    global csv_data

    # Prompt for CSV file input
    file_path = input("Enter the path to the CSV file: ")
    if not file_path.endswith(".csv"):
        print("Invalid file format. Please upload a CSV file.")
        return

    try:
        csv_data = pd.read_csv(file_path)
        print(f"Loaded CSV file: {file_path}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    while True:
        query_input = input("Enter your query (or type 'exit' to quit): ")
        if query_input.lower() == "exit":
            break

        if chatbot_pipeline and csv_data is not None:
            try:
                csv_text = extract_relevant_data(query_input, csv_data)
                response = chatbot_pipeline.invoke(
                    {"question": query_input, "data": csv_text}
                )
                print("\nQuery Result:\n", csv_text)
                print("\nChatbot Response:\n", response)
            except Exception as e:
                print(f"Error during chatbot invocation: {e}")


if __name__ == "__main__":
    main()
