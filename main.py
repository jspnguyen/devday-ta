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

def rag_query(user_query):
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="YZXPI4CEXE",
        retrieval_config={
            "vectorSearchConfiguration": {
                "overrideSearchType": "HYBRID",
            }
        }
    )
    
    rag_results = retriever.retrieve(user_query)
    
    return rag_results

def llm_query(user_query, rag_results):
    # Initialize the LLM with the correct model name and parameters
    llm = Bedrock(model="anthropic.claude-v2", temperature=0, max_tokens=3000)
    response_synthesizer = get_response_synthesizer(
        response_mode="compact", llm=llm
    )

    # Synthesize response
    response_obj = response_synthesizer.synthesize(user_query, rag_results)
    return(response_obj)

if __name__ == "__main__":
    user_query = str(input("Enter your question: "))
    rag_results = rag_query(user_query)
    llm_response = llm_query(user_query, rag_results)
    print(llm_response)
