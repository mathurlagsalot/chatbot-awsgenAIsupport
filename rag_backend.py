import boto3
import json
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import RetrievalQA
# from langchain.llms import Bedrock (while using a different FM)
from langchain.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat

# AWS Region
region_name = "eu-west-1"

# Initialize Amazon Bedrock Clients
bedrock_runtime = boto3.client("bedrock-runtime", region_name=region_name)
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=region_name)

# Initialize Knowledge Base Retriever
kb_id = "LXOGMZS9RJ"
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=kb_id,
    client=bedrock_agent_runtime
)

# Initialize Llama 3 Model
llm = BedrockChat(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    client=bedrock_runtime
)

# Initialize Titan Embeddings
embeddings = BedrockEmbeddings(
    model_id="embed-english-v3.0",
    client=bedrock_runtime

)

# Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)


# AWS Lambda Function
def lambda_handler(event, context):
    """Processes a question using RetrievalQA."""
    try:
        # ✅ Extract Question from Event
        body = json.loads(event["body"])
        question = body.get("question", "").strip()

        if not question:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Please enter a valid question."})
            }

        # Run RAG Pipeline to Get Answer
        answer = qa_chain.run(question)

        # Ensure Proper JSON Response Format
        return {
            "statusCode": 200,
            "body": json.dumps({"answer": answer})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }