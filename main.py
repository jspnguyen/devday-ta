import gradio, llama_index, boto3
from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from llama_index.core import get_response_synthesizer
from llama_index.llms.bedrock.base import Bedrock

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="<knowledge-base-id>",
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 4,
            "overrideSearchType": "HYBRID",
            "filter": {"equals": {"key": "tag", "value": "space"}},
        }
    },
)

query = "How big is Milky Way as compared to the entire universe?"
retrieved_results = retriever.retrieve(query)

# Prints the first retrieved result
print(retrieved_results[0].get_content())

llm = Bedrock(model="anthropic.claude-v2", temperature=0, max_tokens=3000)
response_synthesizer = get_response_synthesizer(
    response_mode="compact", llm=llm
)
response_obj = response_synthesizer.synthesize(query, retrieved_results)
print(response_obj)