import gradio, llama_index, boto3, ldclient, os
from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from llama_index.core import get_response_synthesizer
from llama_index.llms.bedrock.base import Bedrock
from ldclient.config import Config
from dotenv import load_dotenv

# load_dotenv()
# ldclient.set_config(Config(os.getenv("ldl_sdk_key")))
# client = ldclient.get()

try:
    s3_client = boto3.client('s3')
    print("Successfully initialized S3 client.")
except Exception as e:
    print(f"Error initializing S3 client: {e}")

# Initialize the retriever with the correct configuration
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="YZXPI4CEXE",
    retrieval_config={
        "vectorSearchConfiguration": {
            "overrideSearchType": "HYBRID",
        }
    }
)

query = "What is a poisson distribution?"

try:
    # Debugging output to verify input values
    print("Retrieving results with the following parameters:")
    print(f"Knowledge Base ID: {retriever.knowledge_base_id}")
    print(f"Retrieval Config: {retriever.retrieval_config}")
    print(f"Query: {query}")

    retrieved_results = retriever.retrieve(query)

    # Check if any results were retrieved
    if retrieved_results:
        # Prints the first retrieved result
        print(retrieved_results[0].get_content())
        print("--------------------------")
    else:
        print("No results retrieved.")

    # Initialize the LLM with the correct model name and parameters
    llm = Bedrock(model="anthropic.claude-v2", temperature=0, max_tokens=3000)
    response_synthesizer = get_response_synthesizer(
        response_mode="compact", llm=llm
    )

    # Synthesize response
    response_obj = response_synthesizer.synthesize(query, retrieved_results)
    print(response_obj)

except boto3.exceptions.Boto3Error as e:
    print(f"An AWS error occurred: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()