import boto3, ldclient, os
import gradio as gr
from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from llama_index.core import get_response_synthesizer
from llama_index.llms.bedrock.base import Bedrock
from ldclient.config import Config
from dotenv import load_dotenv
from ldclient import Context

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
    context = Context.builder("anonymous").name("anonymous").build()
    search_flag = client.variation("select_search_type", context, True)
    
    retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id="YZXPI4CEXE",
            retrieval_config={
                "vectorSearchConfiguration": {
                    "overrideSearchType": f"{search_flag}",
                }
            }
        )
    
    value_filter = ""
    if "exam" in user_query.lower() or "midterm" in user_query.lower() or "final" in user_query.lower():
        value_filter = "exam"
    elif "homework" in user_query.lower():
        value_filter = "hw"
    elif "note" in user_query.lower():
        value_filter = "note"
    elif "discussion" in user_query.lower():
        value_filter = "disc"
    
    if value_filter:
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id="YZXPI4CEXE",
            retrieval_config={
                "vectorSearchConfiguration": {
                    "overrideSearchType": f"{search_flag}",
                    "filter": {"equals": {"key": "type", "value":f"{value_filter}"}},
                }
            }
        )
    
    rag_results = retriever.retrieve(user_query)
    return rag_results

def llm_query(user_query, rag_results):
    """
    Queries response from LLM using RAG results and current LLM model
    """
    context = Context.builder("anonymous").name("anonymous").build()
    model_flag = client.variation("select_claude_model", context, True)
    
    llm = Bedrock(model=model_flag, temperature=0.1, max_tokens=4096)
    
    response_synthesizer = get_response_synthesizer(
        response_mode="compact", llm=llm
    )

    response_obj = response_synthesizer.synthesize(user_query, rag_results)
    return str(response_obj)

def chat_with_bot(user_input, _):
    """
    Handle chatbot responses with users. Control personality and accuracy of the responses.
    """
    personality = (
        "You are a friendly and knowledgeable CS70 TA. "
        "You are patient, encouraging, and always provide detailed explanations. "
        "You enjoy making learning fun and engaging for students."
        "You adapt your teaching based on the student's understanding of the topic."
    )
    
    formatted_input = f'{personality}\n\nUser prompt:\n{user_input}'
    
    rag_results = rag_query(formatted_input)
    llm_response = llm_query(formatted_input, rag_results)
    src_s3_url = rag_results[0].metadata['sourceMetadata']['x-amz-bedrock-kb-source-uri']
    src_filename = os.path.basename(src_s3_url)
    full_response = llm_response + '\n\n This response was based on https://www.eecs70.org/assets/pdf/' + src_filename + '. For more information check out the source material!'
    
    return full_response

if __name__ == "__main__":
    example_inputs = ["Explain to me Poisson distributions.", "Ask me a practice question from a past exam.", "What topics should I review for the final?"]
    interface = gr.ChatInterface(
            fn=chat_with_bot, 
            examples=example_inputs,
            title="GradeGuardian",
            description="Ask the GradeGuardian any question you want about CS70 or use it to quiz yourself!",
            theme=gr.themes.Soft(),
            submit_btn="Ask",
            stop_btn="Stop"
    )
    
    interface.launch(share=True)
