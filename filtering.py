# %%
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

def rag_query(user_query, key='course', val='cs70'):
    """
    Queries appropriate documents from AWS knowledge bases using Hybrid RAG vector search
    """
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="YZXPI4CEXE",
        retrieval_config={
            "vectorSearchConfiguration": {
                "overrideSearchType": "HYBRID",
                "filter": {"equals": {"key": key, "value": val}},
            }
        }
    )
    
    rag_results = retriever.retrieve(user_query)
    return rag_results

def llm_query(user_query, rag_results):
    """
    Queries response from LLM using RAG results and current LLM model
    """
    llm = Bedrock(model="anthropic.claude-3-sonnet-20240229-v1:0", temperature=0.1, max_tokens=4096)
    
    response_synthesizer = get_response_synthesizer(
        response_mode="compact", llm=llm
    )

    response_obj = response_synthesizer.synthesize(user_query, rag_results)
    return str(response_obj)

# %%
outputs = rag_query('what is a poisson distribution?')

# %%
src = outputs[0].metadata['sourceMetadata']['x-amz-bedrock-kb-source-uri']
filename = os.path.basename(src)
filename

# %%
def chat_with_bot(user_input, key, val):
    """
    Handle chatbot responses with users. Control personality and accuracy of the responses.
    """
    personality = (
        "You are a friendly and knowledgeable CS70 TA. "
        "You are patient, encouraging, and always provide detailed explanations. "
        "You enjoy making learning fun and engaging for students."
    )
    
    formatted_input = f'{personality}\n\nUser prompt:\n{user_input}'
    
    rag_results = rag_query(formatted_input, key, val)
    llm_response = llm_query(formatted_input, rag_results)
    src_s3_url = rag_results[0].metadata['sourceMetadata']['x-amz-bedrock-kb-source-uri']
    src_filename = os.path.basename(src_s3_url)
    full_response = llm_response + '\n This response was based on https://www.eecs70.org/assets/pdf/' + src_filename + '. For more information check out the source material!'
    return full_response

# %%
example_inputs = ["Explain to me Poisson distributions", "Ask me a practice question from a past exam", "What is CS70?"]

with gr.Blocks() as demo:
    disc = gr.Button("Discussions")
    exams = gr.Button("Exams")
    all = gr.Button('Query from Everything')
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    

    def respond(user_input, chat_history):
        bot_text = chat_with_bot(user_input, Filter.key, Filter.val)
        chat_history.append((user_input, bot_text))
        return ("", chat_history)
        
    def filterset(key, val):
        Filter.key = key
        Filter.val = val
        chatbot.label='now querying from '+val

    class Filter:
        key = None
        val = None
    

    disc.click(filterset('type', 'disc'), None, chatbot)
    exams.click(filterset('type', 'exam'))
    all.click(filterset('course', 'cs70'))

    msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

    
demo.queue()
demo.launch()



# %%
interface = gr.ChatInterface(
        fn=chat_with_bot, 
        examples=example_inputs,
        title="CS70 TA",
        description="Ask the TA any question you want about CS70 or use it to quiz yourself!",
        theme=gr.themes.Soft(),
        submit_btn="Ask",
        stop_btn="Stop stream"
    )

interface.launch(share=False)

# %%
