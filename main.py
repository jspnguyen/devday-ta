import boto3, ldclient, os
import gradio as gr
from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from llama_index.core import get_response_synthesizer
from llama_index.llms.bedrock.base import Bedrock
from ldclient.config import Config
from dotenv import load_dotenv

try:
    load_dotenv()
    ldclient.set_config(Config(os.getenv("ldl_sdk_key")))
    client = ldclient.get()
    print("Successfully initialized LDL client.")
    s3_client = boto3.client('s3')
    print("Successfully initialized S3 client.")
except Exception as e:
    print(f"Error initializing S3 client: {e}")

def rag_query(user_query):
    """
    Queries appropriate documents from AWS knowledge bases using Hybrid RAG vector search
    """
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
    """
    Queries response from LLM using RAG results and current LLM model
    """
    llm = Bedrock(model="anthropic.claude-v2", temperature=0, max_tokens=3000)
    
    response_synthesizer = get_response_synthesizer(
        response_mode="compact", llm=llm
    )

    response_obj = response_synthesizer.synthesize(user_query, rag_results)
    return str(response_obj)

def chat_with_bot(user_input, _):
    rag_results = rag_query(user_input)
    llm_response = llm_query(user_input, rag_results)
    
    return llm_response

if __name__ == "__main__":
    interface = gr.ChatInterface(
            fn=chat_with_bot, 
            # examples=example_inputs,
            title="CS70 TA",
            description="Ask the TA any question you want about CS70 or use it to quiz yourself!",
            theme=gr.themes.Soft(),
            submit_btn="Ask",
            stop_btn="Stop stream"
        )
    
    interface.launch(share=False)
