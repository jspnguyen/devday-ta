import gradio, llama_index, boto3, ldclient, os
from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from llama_index.core import get_response_synthesizer
from llama_index.llms.bedrock.base import Bedrock
from ldclient.config import Config
from dotenv import load_dotenv

s3_client = boto3.client('s3')

# load_dotenv()
# ldclient.set_config(Config(os.getenv("ldl_sdk_key")))
# client = ldclient.get()

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="YZXPI4CEXE",
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 4,
            "overrideSearchType": "HYBRID",
            "filter": {"equals": {"key": "tag", "value": "space"}},
        }
    },
)

query = """
User prompt: 
What is a poisson distribution?
"""

# ! Line below is erroring out, please save us!
retrieved_results = retriever.retrieve(query)

# Prints the first retrieved result
print(retrieved_results[0].get_content())
print("--------------------------")

llm = Bedrock(model="anthropic.claude-v2", temperature=0, max_tokens=3000)
response_synthesizer = get_response_synthesizer(
    response_mode="compact", llm=llm
)
response_obj = response_synthesizer.synthesize(query, retrieved_results)
print(response_obj)

# if __name__ == "__main__":
#     main()