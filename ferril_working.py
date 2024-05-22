from distutils.command.config import config
import gradio, llama_index, boto3
from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from llama_index.core import get_response_synthesizer
from llama_index.llms.bedrock.base import Bedrock
from botocore.client import Config
from llama_index.core.llms import ChatMessage

import os
import ldclient
from dotenv import load_dotenv
from ldclient import Context
from ldclient.config import Config
from threading import Lock, Event


# Set sdk_key to your LaunchDarkly SDK key.
load_dotenv()
sdk_key = os.getenv("LAUNCHDARKLY_SDK_KEY")
ldclient.set_config(Config(sdk_key))
client = ldclient.get()

# Set feature_flag_key to the feature flag key you want to evaluate.
feature_flag_key = "sample-feature"


def show_evaluation_result(key: str, value: bool):
    print()
    print(f"*** The {key} feature flag evaluates to {value}")


def show_banner():
    print()
    print("        ██       ")
    print("          ██     ")
    print("      ████████   ")
    print("         ███████ ")
    print("██ LAUNCHDARKLY █")
    print("         ███████ ")
    print("      ████████   ")
    print("          ██     ")
    print("        ██       ")
    print()


class FlagValueChangeListener:
    def __init__(self):
        self.__show_banner = True
        self.__lock = Lock()

    def flag_value_change_listener(self, flag_change):
        with self.__lock:
            if self.__show_banner and flag_change.new_value:
                show_banner()
                self.__show_banner = False

            show_evaluation_result(flag_change.key, flag_change.new_value)


if __name__ == "__main__":
    if not sdk_key:
        print("*** Please set the LAUNCHDARKLY_SDK_KEY env first")
        exit()
    if not feature_flag_key:
        print("*** Please set the LAUNCHDARKLY_FLAG_KEY env first")
        exit()

    ldclient.set_config(Config(sdk_key))

    if not ldclient.get().is_initialized():
        print("*** SDK failed to initialize. Please check your internet connection and SDK credential for any typo.")
        exit()

    print("*** SDK successfully initialized")

    # Set up the evaluation context. This context should appear on your
    # LaunchDarkly contexts dashboard soon after you run the demo.
    # context = \
    #     Context.builder('example-user-key').kind('user').name('Sandy').build()

    # flag_value = ldclient.get().variation(feature_flag_key, context, False)
    # show_evaluation_result(feature_flag_key, flag_value)

    # change_listener = FlagValueChangeListener()
    # listener = ldclient.get().flag_tracker \
    #     .add_flag_value_change_listener(feature_flag_key, context, change_listener.flag_value_change_listener)

    # try:
    #     Event().wait()
    # except KeyboardInterrupt:
    #     pass
    # bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts':0})
    # bedrock_client = boto3.client('bedrock-runtime')
    # bedrock_agent_client = boto3.client('bedrock-agent-runtime',
    #                                     config=bedrock_config,
    #                                     region_name="us-west-2"

    # )

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="YZXPI4CEXE",
        retrieval_config={
            "vectorSearchConfiguration": {
                "overrideSearchType": "HYBRID",
            }
        },
    )

    query = "Explain chinese remainder theorem and an example question"
    retrieved_results = retriever.retrieve(query)

    # Prints the first retrieved result
    print(retrieved_results[0].get_content())

    context = Context.builder("anonymous").name("anonymous").build()
    flag_value = client.variation("modelSelection2", context, False)
    print(flag_value)


    llm = Bedrock(model=flag_value, temperature=0, max_tokens=3000)
    response_synthesizer = get_response_synthesizer(
        response_mode="compact", llm=llm
    )
    response_obj = response_synthesizer.synthesize(query, retrieved_results)
    print(response_obj)