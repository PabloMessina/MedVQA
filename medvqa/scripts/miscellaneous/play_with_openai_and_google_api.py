import argparse
import logging
from medvqa.utils.logging_utils import setup_logging
from medvqa.utils.openai_api_utils import run_common_boilerplate_for_api_requests

# Set up logging
setup_logging(log_level=logging.DEBUG)

def parse_output(output):
    return output # This is a placeholder function. You can implement your own parsing logic here.

def mini_test_1(args):
    """Summarize a simple sentence using OpenAI."""
    texts = [
        "Artificial intelligence is transforming the world in many exciting ways."
    ]
    system_instructions = "Summarize the following text in one sentence."
    run_common_boilerplate_for_api_requests(
        texts=texts,
        system_instructions=system_instructions,
        model_name="gpt-3.5-turbo",
        api_key_name="OPENAI_API_KEY_1",
        api_type="openai",
        max_requests_per_minute=100,
        max_tokens_per_minute=10000,
        parse_output=parse_output,
        delete_api_requests_and_responses=args.delete_api_requests_and_responses,
        save_filepath="tmp/openai/summary_results.jsonl",
    )

def mini_test_2(args):
    """Translate a sentence to French using OpenAI."""
    texts = [
        "The quick brown fox jumps over the lazy dog."
    ]
    system_instructions = "Translate the following text to French."
    run_common_boilerplate_for_api_requests(
        texts=texts,
        system_instructions=system_instructions,
        model_name="gpt-3.5-turbo",
        api_key_name="OPENAI_API_KEY_1",
        api_type="openai",
        max_requests_per_minute=100,
        max_tokens_per_minute=10000,
        parse_output=parse_output,
        delete_api_requests_and_responses=args.delete_api_requests_and_responses,
        save_filepath="tmp/openai/translation_results.jsonl",
    )

def mini_test_gemini(args):
    """Summarize a sentence using Google Gemini API."""
    texts = [
        "Artificial intelligence is transforming the world in many exciting ways."
    ]
    system_instructions = "Summarize the following text in one sentence."
    run_common_boilerplate_for_api_requests(
        texts=texts,
        system_instructions=system_instructions,
        model_name="gemini-2.0-flash",
        api_key_name="GEMINI_API_KEY",
        api_type="gemini",
        max_requests_per_minute=100,
        max_tokens_per_minute=10000,
        parse_output=parse_output,
        delete_api_requests_and_responses=args.delete_api_requests_and_responses,
        save_filepath="tmp/gemini/gemini_results.jsonl",
    )

def main():
    parser = argparse.ArgumentParser(
        description="OpenAI/Google API Request Script"
    )
    parser.add_argument(
        "--test_case",
        type=str,
        required=True,
        choices=["mini_test_1", "mini_test_2", "mini_test_gemini"],
        help="Which test case to run"
    )
    parser.add_argument(
        "--delete_api_requests_and_responses",
        action="store_true",
        help="Delete API requests and responses after execution"
    )
    args = parser.parse_args()

    # Map test case names to functions
    test_cases = {
        "mini_test_1": mini_test_1,
        "mini_test_2": mini_test_2,
        "mini_test_gemini": mini_test_gemini,
    }

    # Run the selected test case
    test_cases[args.test_case](args)

if __name__ == "__main__":
    main()
