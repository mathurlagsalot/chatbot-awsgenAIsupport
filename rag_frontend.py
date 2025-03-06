import streamlit as st
import json
from rag_backend import lambda_handler  # Import AWS Lambda backend function

# Streamlit App Title
st.title("RAG Application with Amazon Bedrock")

# User Input
question = st.text_input("Enter your question:")

# Button to Fetch Answer
if st.button("Get Answer"):
    if question.strip():  # Ensures input is not just whitespace
        try:
            # ✅ Correctly Format Input as JSON
            event = {"body": json.dumps({"question": question})}

            # ✅ Call Lambda Handler Function
            response = lambda_handler(event, None)

            # ✅ Parse JSON Response from Lambda
            response_body = json.loads(response["body"])

            # ✅ Display Answer if Present
            if "answer" in response_body and response_body["answer"]:
                st.success("Answer:")
                st.write(response_body["answer"])
            elif "error" in response_body:
                st.error(f"Error: {response_body['error']}")
            else:
                st.warning("No answer received from the backend.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")  # Handle Backend Errors
    else:
        st.warning("Please enter a valid question.")  # Warning for Empty Input
