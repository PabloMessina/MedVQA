# Taken and adapted from https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py

# imports
import math
import sys
from openai import OpenAI, AsyncOpenAI # for OpenAI API calls
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
import random  # for sampling error messages
from dataclasses import dataclass, field

from medvqa.utils.common import get_timestamp
from medvqa.utils.files_utils import load_jsonl, load_pickle, save_jsonl, save_pickle
from medvqa.utils.logging_utils import ANSI_MAGENTA_BOLD, ANSI_RESET  # for storing API inputs, outputs, and metadata

logger = logging.getLogger(__name__)

__all__ = [
    "process_api_requests_from_file",
    "GPT_IS_ACTING_WEIRD_REGEX",
    "run_common_boilerplate_for_api_requests",
]

ALLOWED_MODELS = {
    "openai": set([
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-1106-preview",
        "gpt-4o",
        "gpt-4o-mini",
        # ...add more as released
    ]),
    "gemini": set([
        "gemini-2.0-pro",
        "gemini-2.0-flash",
        "gemini-2.5-pro-exp-03-25",
        "gemini-2.5-pro-preview-03-25",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-flash-preview-04-17-thinking",
        "gemini-2.5-pro-preview-05-06",
        # ...add more as released
    ]),
}

REASONING_MODELS = {
    "gemini": set([
        "gemini-2.5-pro-exp-03-25",
        "gemini-2.5-pro-preview-03-25",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-flash-preview-04-17-thinking",
        "gemini-2.5-pro-preview-05-06",
    ]),
    "openai": set([
        "gpt-4o",
        "gpt-4o-mini",
    ]),
}

GPT_IS_ACTING_WEIRD_REGEX = re.compile(r"\b(I'm sorry|Sorry|Could you|Can you|I apologize|Sure|I'd be happy|do you need)\b")


def process_api_requests_from_file(
    requests_filepath: str,
    api_type: str,
    api_key: str,
    save_filepath: str = None,
    max_requests_per_minute: float = 3_000 * 0.5,
    max_tokens_per_minute: float = 250_000 * 0.5,
    token_encoding_name: str = None,
    max_attempts: int = 3,
    log_info_every_n_requests: int = 100,
    base_url: str = None, # Add base_ur l as a parameter with a default of None
):
    """Processes API requests in parallel from a file, throttling to stay under rate limits.
    Can target either OpenAI or Gemini compatibility API based on base_url.
    """
    assert os.path.exists(requests_filepath), f"API requests file {requests_filepath} does not exist"
    assert api_key is not None, "API key must be provided. Set GOOGLE_API_KEY or OPENAI_API_KEY environment variable or pass it directly."
    assert max_requests_per_minute is not None, "Max requests per minute must be provided"
    assert max_tokens_per_minute is not None, "Max tokens per minute must be provided"
    assert max_attempts is not None, "Max attempts must be provided"
    assert log_info_every_n_requests is not None, "Log info every n requests must be provided"

    if save_filepath is None:
        save_filepath = requests_filepath.replace(".jsonl", "_results.jsonl")
        logger.info(f"Saving results to {save_filepath}")

    # We will handle token counting later, so token_encoding_name is not passed down directly
    # You might want to pass the model name down to _process_api_requests_from_file
    # if you need model-specific token counting or other logic there.

    return asyncio.run(
        _process_api_requests_from_file(
            requests_filepath=requests_filepath,
            save_filepath=save_filepath,
            api_type=api_type,
            api_key=api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=max_attempts,
            log_info_every_n_requests=log_info_every_n_requests,
            base_url=base_url, # Pass the new base_url parameter
        )
    )


def is_model_supported(model_name, api_type):
    """Check if the model name is supported for the given API type."""
    if api_type == "openai":
        return model_name in ALLOWED_MODELS["openai"]
    elif api_type == "gemini":
        return model_name in ALLOWED_MODELS["gemini"]
    else:
        raise ValueError(f"Unknown API type: {api_type}. Must be 'openai' or 'gemini'.")


def _generate_request(
    system_instructions: str,
    query: str,
    model_name: str,
    max_tokens: int,
    temperature: float = 0,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    api_type: str = "openai",  # 'openai' or 'gemini'
):
    assert len(system_instructions) > 0
    assert len(query) > 0

    request = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": query},
        ],
        "metadata": {"query": query},
    }

    # Only include parameters supported by the API
    if api_type == "openai":
        request["temperature"] = temperature # 0.0 = (almost) deterministic, 2.0 = max entropy
        request["frequency_penalty"] = frequency_penalty # 0.0 = no penalty, 2.0 = max penalty
        request["presence_penalty"] = presence_penalty # 0.0 = no penalty, 2.0 = max penalty
        request["max_tokens"] = max_tokens
    elif api_type == "gemini":
        # Gemini supports temperature and max_tokens, but not frequency_penalty or presence_penalty
        request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        # Do NOT include frequency_penalty or presence_penalty
    else:
        raise ValueError(f"Unknown API type: {api_type}. Must be 'openai' or 'gemini'.")

    return request


# Based on https://platform.openai.com/docs/guides/batch/getting-started
def _generate_batch_request(custom_id, model_name, system_instructions, query, max_tokens,
                            temperature=0.0, frequency_penalty=0.0, presence_penalty=0.0):
    assert len(system_instructions) > 0
    assert len(query) > 0
    assert is_model_supported(model_name, "openai"), f"Unknown model name: {model_name} for API openai"
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_instructions,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
        },
    }

def run_common_boilerplate_for_api_requests(
    api_responses_filepath: str = None,
    texts: list[str] = None,
    system_instructions: str = None,
    api_key_name: str = "OPENAI_API_KEY", # Name of the environment variable for the API key
    model_name: str = None, # Use a more general model_name parameter
    max_tokens_per_request: int = None,
    max_requests_per_minute: float = None,
    max_tokens_per_minute: float = None,
    temperature: float = 0,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    parse_output: callable = None,
    tmp_dir: str = "tmp",
    save_filepath: str = None,
    delete_api_requests_and_responses: bool = True,
    use_batch_api: bool = False, # OpenAI-specific batch API flag
    batch_description: str = None,
    batch_input_file_id: str = None,
    api_type: str = "openai", # New parameter: 'openai' or 'gemini'
    gemini_api_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/", # Default for Gemini compatibility
    log_info_every_n_requests: int = 50, # Add log_info_every_n_requests back
):
    """Runs common boilerplate for API requests.

    Handles both direct API calls (throttled) and OpenAI Batch API.
    Supports targeting OpenAI or Gemini compatibility API based on base_url.
    """

    if api_responses_filepath is None:

        if use_batch_api:
            
            if batch_input_file_id is not None:

                assert parse_output is not None, "parse_output must be provided when batch_input_file_id is provided"

                batch_object_filepath = os.path.join(tmp_dir, "openai", "batch_api", f"{batch_input_file_id}.pkl")
                assert os.path.exists(batch_object_filepath), f"Batch object file {batch_object_filepath} does not exist"
                logger.info(f"Checking status of batch with input file ID {batch_input_file_id}")
                
                batch_object = load_pickle(batch_object_filepath)
                logger.info(f"Batch object: {batch_object}")
                
                # We need to check the status of an existing batch already created through the API
                client = OpenAI(api_key=os.getenv(api_key_name))
                batch = client.batches.retrieve(batch_object.id)
                logger.info(f"Batch: {batch}")
                logger.info(f"Batch status: {batch.status}")

                if batch.output_file_id is None:
                    logger.warning(f"Batch output file ID is None. Exiting.")
                    sys.exit(1)
                
                if batch.status in ['completed', 'cancelled']:
                    logger.info(f"Batch with input file ID {batch_input_file_id} is {batch.status}")

                    logger.info(f"Retrieving batch input file ID {batch.input_file_id}")
                    input_content = client.files.content(batch.input_file_id)
                    custom_id_to_metadata = {}
                    for line in input_content.iter_lines():
                        api_request = json.loads(line)
                        custom_id = api_request['custom_id']
                        query = api_request['body']['messages'][1]['content']
                        metadata = { "query": query }
                        custom_id_to_metadata[custom_id] = metadata
                    logger.info(f"Retrieved metadata for {len(custom_id_to_metadata)} queries")
                    
                    logger.info(f"Retrieving batch output file ID {batch.output_file_id}")
                    output_content = client.files.content(batch.output_file_id)
                    postprocessed_responses = []
                    api_responses = []
                    error_messages = []
                    for line in output_content.iter_lines():
                        api_response = json.loads(line)
                        api_responses.append(api_response)
                        try:
                            # text = api_response[1]['choices'][0]['message']['content']
                            custom_id = api_response['custom_id']
                            text = api_response['response']['body']['choices'][0]['message']['content']
                            metadata = custom_id_to_metadata[custom_id]
                            parsed_output = parse_output(text)
                            postprocessed_responses.append({
                                "metadata": metadata,
                                "parsed_response": parsed_output,
                            })
                        except Exception as e:
                            api_response_string = json.dumps(api_response)
                            error_messages.append(f"Error parsing response {api_response_string} for query \"{metadata['query']}\": {e}")
                            continue
                    if len(error_messages) > 0:
                        logger.error(f"{len(error_messages)} error messages occurred while parsing responses")
                        # sample 5 error messages
                        for i in random.sample(range(len(error_messages)), min(5, len(error_messages))):
                            logger.error(error_messages[i])
                    if len(postprocessed_responses) == 0:
                        logger.warning(f"None of the {len(api_responses)} API responses could be parsed. Exiting.")
                    else:
                        # Save processed texts by appending to existing file
                        n_processed = len(postprocessed_responses)
                        n_total = len(api_responses)
                        logger.info(f"""Succesfully processed {n_processed} of {n_total} API responses.
                                    {n_total - n_processed} of {n_total} API responses could not be processed.
                                    Saving processed texts to {save_filepath}""")
                        save_jsonl(postprocessed_responses, save_filepath, append=True)

                return # We are done with the batch API
            
            else:
                # Check if batch_description is provided
                assert batch_description is not None, "batch_description must be provided when use_batch_api is True"
                
                # Prepare API requests
                requests = []
                for i, text in enumerate(texts):
                    requests.append(_generate_batch_request(
                        custom_id=str(i),
                        model_name=model_name,
                        system_instructions=system_instructions,
                        query=text,
                        max_tokens=max_tokens_per_request,
                        temperature=temperature,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                    ))
                timestamp = get_timestamp()
                batch_api_requests_filepath = os.path.join(tmp_dir, "openai", f"batch_api_requests_{timestamp}.jsonl")
                logger.info(f"Saving batch API requests to {batch_api_requests_filepath}")
                save_jsonl(requests, batch_api_requests_filepath)
                
                client = OpenAI(api_key=os.getenv(api_key_name))
                batch_input_file = client.files.create(
                    file=open(batch_api_requests_filepath, "rb"),
                    purpose="batch"
                )
                batch_input_file_id = batch_input_file.id
                logger.info(f"Creating batch with input file ID {batch_input_file_id}")

                batch_object =client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": batch_description,
                    },
                )
                logger.info(f"Batch object: {batch_object}")
                batch_object_filepath = os.path.join(tmp_dir, "openai", "batch_api", f"{batch_input_file_id}.pkl")
                logger.info(f"Saving batch object to {batch_object_filepath}")
                save_pickle(batch_object, batch_object_filepath)

                # Delete batch API requests
                if delete_api_requests_and_responses:
                    logger.info(f"Deleting batch API requests at {batch_api_requests_filepath}")
                    os.remove(batch_api_requests_filepath)

                return # Exit early, because batch API requests are asynchronous
        
        else:
            # Determine the base_url based on api_type
            if api_type == "openai":
                # Use the default OpenAI base URL (handled by the client when base_url is None)
                target_base_url = None
                token_encoding_for_processing = tiktoken.encoding_for_model(model_name).name

            elif api_type == "gemini":
                # Use the specified gemini_api_base_url for Gemini compatibility
                target_base_url = gemini_api_base_url
                # Tiktoken won't work for Gemini
                token_encoding_for_processing = None # Tiktoken is not applicable

            else:
                raise ValueError(f"Unknown api_type: {api_type}. Must be 'openai' or 'gemini'.")

            # Prepare API requests for direct calls
            requests = []
            for text in texts:
                # Use the potentially updated _generate_request that handles different model names
                requests.append(_generate_request(
                    system_instructions=system_instructions,
                    query=text,
                    model_name=model_name, # Use the general model_name
                    api_type=api_type, # Pass the api_type
                    max_tokens=max_tokens_per_request,
                    temperature=temperature,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                ))
                assert 'metadata' in requests[-1] # Ensure metadata is included

            timestamp = get_timestamp()
            # Use api_type in the filename to distinguish request files
            api_requests_filepath_generated = os.path.join(tmp_dir, api_type, f"api_requests_{timestamp}.jsonl")
            api_responses_filepath_generated = os.path.join(tmp_dir, api_type, f"api_responses_{timestamp}.jsonl")

            logger.info(f"Saving API requests to {api_requests_filepath_generated}")
            logger.info(f"Saving API responses to {api_responses_filepath_generated}")
            save_jsonl(requests, api_requests_filepath_generated)

            # Send API requests using the throttled processor
            process_api_requests_from_file(
                requests_filepath=api_requests_filepath_generated,
                save_filepath=api_responses_filepath_generated,
                api_type=api_type,
                api_key=os.getenv(api_key_name), # Use the provided api_key_name
                max_requests_per_minute=max_requests_per_minute,
                max_tokens_per_minute=max_tokens_per_minute,
                token_encoding_name=token_encoding_for_processing, # Pass the determined encoding name (or None)
                max_attempts=5,
                log_info_every_n_requests=log_info_every_n_requests,
                base_url=target_base_url, # Pass the determined base_url
            )
            # Update api_responses_filepath to the generated file for postprocessing
            api_responses_filepath = api_responses_filepath_generated
            api_requests_filepath = api_requests_filepath_generated # Keep track of the requests file for potential deletion

    else:
        assert os.path.exists(api_responses_filepath), f"API responses file {api_responses_filepath} does not exist"
        logger.info(f"API responses already exist at {api_responses_filepath}. Skipping API requests.")
        dir_path = os.path.dirname(api_responses_filepath)
        filename = os.path.basename(api_responses_filepath)
        api_requests_filepath = os.path.join(dir_path, filename.replace("responses", "requests"))
        assert os.path.exists(api_requests_filepath), f"API requests file {api_requests_filepath} does not exist"
        requests = None

    assert parse_output is not None, "parse_output must be provided"
    assert save_filepath is not None, "save_filepath must be provided"

    # Load and postprocess API responses
    logger.info(f"Loading API responses from {api_responses_filepath}")
    api_responses = load_jsonl(api_responses_filepath)
    if requests is not None:
        assert len(api_responses) == len(requests)

    postprocessed_responses = []

    for api_response in api_responses:
        # Expecting a dict with keys: request, response, metadata, errors
        metadata = api_response.get("metadata")
        response = api_response.get("response")
        errors = api_response.get("errors", [])
        if response is None or errors:
            logger.error(f"Skipping response due to errors: {errors}")
            continue
        try:
            text = response['choices'][0]['message']['content']
            parsed_output = parse_output(text)
            postprocessed_responses.append({
                "metadata": metadata,
                "parsed_response": parsed_output,
            })
        except Exception as e:
            api_response_string = json.dumps(api_response)
            if len(api_response_string) > 500:
                api_response_string = api_response_string[:250] + "..." + api_response_string[-250:]
            logger.error(f"Error parsing response {api_response_string} for query \"{metadata['query']}\": {e}")
            continue

    # Delete API requests and responses
    if delete_api_requests_and_responses:
        logger.info(f"Deleting API requests and responses")
        os.remove(api_requests_filepath)
        os.remove(api_responses_filepath)

    if len(postprocessed_responses) == 0:
        logger.warning(f"None of the {len(api_responses)} API responses could be parsed. Exiting.")
    else:
        # Save processed texts by appending to existing file
        n_processed = len(postprocessed_responses)
        n_total = len(api_responses)
        logger.info(f"Succesfully processed {n_processed} of {n_total} API responses. "
                    f"{n_total - n_processed} of {n_total} API responses could not be processed. "
                    f"Saving processed texts to {ANSI_MAGENTA_BOLD}{save_filepath}{ANSI_RESET}")
        save_jsonl(postprocessed_responses, save_filepath, append=True)



async def _process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    api_type: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    max_attempts: int,
    log_info_every_n_requests: int = 100,
    base_url: str = None,
    token_encoding_name: str = None,
):
    """Processes API requests in parallel, throttling to stay under rate limits.
    Can target either OpenAI or Gemini compatibility API based on base_url."""
    
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # Initialize the OpenAI client
    try:
        # If base_url is provided, use it. Otherwise, the client will use the default OpenAI base URL.
        if base_url:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
            logger.debug(f"OpenAI client initialized with base_url: {base_url}")
        else:
            client = AsyncOpenAI(
                api_key=api_key
            )
            logger.debug("OpenAI client initialized with default OpenAI base URL.")

    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logger.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logger.debug(f"File opened. Entering main loop")

        while True:
            # get next request (if one is not already waiting for capacity)
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                    logger.debug(f"Retrying request {next_request.task_id}: {next_request}")
                elif file_not_finished:
                    try:
                        # get new request
                        request_json = json.loads(next(requests))
                        api_endpoint = infer_api_endpoint(request_json)
                        token_consumption = num_tokens_consumed_from_request(
                            request_json, api_endpoint, api_type, token_encoding_name
                        )
                        logger.debug(f"api_endpoint: {api_endpoint}")
                        logger.debug(f"token_consumption: {token_consumption}")
                        next_request = APIRequest(
                            api_type=api_type,
                            task_id=next(task_id_generator),
                            request_json=request_json,
                            token_consumption = token_consumption,
                            attempts_left=max_attempts,
                            metadata=request_json.pop("metadata", None)
                        )
                        status_tracker.num_tasks_started += 1
                        status_tracker.num_tasks_in_progress += 1
                        logger.debug(f"Reading request {next_request.task_id}: {next_request}")
                    except StopIteration:
                        # if file runs out, set flag to stop reading it
                        logger.debug("Read file exhausted")
                        file_not_finished = False

            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity = min(
                available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
                max_requests_per_minute,
            )
            available_token_capacity = min(
                available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
                max_tokens_per_minute,
            )
            last_update_time = current_time

            # if enough capacity available, call API
            if next_request:
                # We'll need a more dynamic way to get token consumption
                next_request_tokens = next_request.token_consumption # This needs to be updated
                if (
                    available_request_capacity >= 1
                    and available_token_capacity >= next_request_tokens
                ):
                    # update counters
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens
                    next_request.attempts_left -= 1

                    # call API
                    asyncio.create_task(
                        next_request.call_api(
                            client, # Pass the initialized client
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker,
                            log_info=status_tracker.num_api_calls % log_info_every_n_requests == 0,
                        )
                    )
                    status_tracker.num_api_calls += 1
                    next_request = None  # reset next_request to empty

            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress == 0:
                break

            # main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(seconds_to_sleep_each_loop)

            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
            if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
                await asyncio.sleep(remaining_seconds_to_pause)
                # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                logger.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

        # after finishing, log final status
        logger.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
        if status_tracker.num_tasks_failed > 0:
            logger.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}.")
        if status_tracker.num_rate_limit_errors > 0:
            logger.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_calls: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    api_type: str  # "openai" or "gemini"
    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        client: AsyncOpenAI,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
        log_info: bool = True
    ):
        """Calls the OpenAI API and saves results."""
        if log_info:
            logger.info(f"Starting request #{self.task_id}")
        error = None
        response = None
        try:
            # Build kwargs based on API type
            kwargs = {
                "model": self.request_json["model"],
                "messages": self.request_json["messages"],
            }
            if self.api_type == "openai":
                kwargs["temperature"] = self.request_json.get("temperature", 0)
                kwargs["frequency_penalty"] = self.request_json.get("frequency_penalty", 0)
                kwargs["presence_penalty"] = self.request_json.get("presence_penalty", 0)
                kwargs["max_tokens"] = self.request_json.get("max_tokens")
            elif self.api_type == "gemini":
                kwargs["temperature"] = self.request_json.get("temperature", 0)
                if self.request_json["model"] in REASONING_MODELS[self.api_type]:
                    kwargs["reasoning_effort"] = self.request_json.get("reasoning_effort", "none")
                if self.request_json.get("max_tokens") is not None:
                    kwargs["max_tokens"] = self.request_json.get("max_tokens")
                # Do NOT include frequency_penalty or presence_penalty
            else:
                raise ValueError(f"Unknown API type: {self.api_type}. Must be 'openai' or 'gemini'.")

            response = await client.chat.completions.create(**kwargs)
            
        except Exception as e:
            logger.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e

        # Error handling
        if error is not None or (response is not None and hasattr(response, 'error')):
            if error is None:
                error = response.error
                logger.warning(f"Request {self.task_id} failed with API error {error}")
                status_tracker.num_api_errors += 1
                if "Rate limit" in str(error):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1

            self.result.append(str(error))
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logger.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                data = {
                    "request": self.request_json,
                    "response": None,
                    "metadata": self.metadata,
                    "errors": self.result,
                }
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            # Successful response
            # Convert response to dict (from Pydantic model)
            response_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            data = {
                "request": self.request_json,
                "response": response_dict,
                "metadata": self.metadata,
                "errors": [],
            }
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logger.debug(f"Request {self.task_id} succeeded and saved to {save_filepath}")


# functions

def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search('^https://[^/]+/v\\d+/(.+)$', request_url)
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def infer_api_endpoint(request_json: dict) -> str:
    if "messages" in request_json:
        return "chat/completions"
    elif "prompt" in request_json:
        return "completions"
    elif "input" in request_json:
        return "embeddings"
    else:
        raise ValueError("Cannot infer API endpoint from request_json")
    

def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    api_type: str,
    token_encoding_name: str = None,
):
    if api_type == "openai":
        assert token_encoding_name is not None
        encoding = tiktoken.get_encoding(token_encoding_name)

        # if completions request, tokens = prompt + n * max_tokens
        if api_endpoint.endswith("completions"):
            max_tokens = request_json.get("max_tokens", 15)
            n = request_json.get("n", 1)
            completion_tokens = n * max_tokens

            # chat completions
            if api_endpoint.startswith("chat/"):
                num_tokens = 0
                for message in request_json["messages"]:
                    num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                    for key, value in message.items():
                        num_tokens += len(encoding.encode(value))
                        if key == "name":  # if there's a name, the role is omitted
                            num_tokens -= 1  # role is always required and always 1 token
                num_tokens += 2  # every reply is primed with <im_start>assistant
                return num_tokens + completion_tokens
            # normal completions
            else:
                prompt = request_json["prompt"]
                if isinstance(prompt, str):  # single prompt
                    prompt_tokens = len(encoding.encode(prompt))
                    num_tokens = prompt_tokens + completion_tokens
                    return num_tokens
                elif isinstance(prompt, list):  # multiple prompts
                    prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                    num_tokens = prompt_tokens + completion_tokens * len(prompt)
                    return num_tokens
                else:
                    raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
        # if embeddings request, tokens = input tokens
        elif api_endpoint == "embeddings":
            input = request_json["input"]
            if isinstance(input, str):  # single input
                num_tokens = len(encoding.encode(input))
                return num_tokens
            elif isinstance(input, list):  # multiple inputs
                num_tokens = sum([len(encoding.encode(i)) for i in input])
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
        # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
        else:
            raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')
    
    elif api_type == "gemini":
        # Gemini API does not use tiktoken, so we need to handle it differently
        # Use a rough estimate: 1 token ≈ 4 characters
        def estimate_tokens(text):
            if not text:
                return 0
            return max(1, math.ceil(len(text) / 4))
        if api_endpoint.endswith("completions"):
            n = request_json.get("n", 1)
            max_tokens = request_json.get("max_tokens", 15)
            completion_tokens = n * max_tokens
            if api_endpoint.startswith("chat/"):
                num_tokens = 0
                for message in request_json["messages"]:
                    for key, value in message.items():
                        num_tokens += estimate_tokens(value)
                return num_tokens + completion_tokens
            else:
                prompt = request_json["prompt"]
                if isinstance(prompt, str):
                    prompt_tokens = estimate_tokens(prompt)
                    return prompt_tokens + completion_tokens
                elif isinstance(prompt, list):
                    prompt_tokens = sum([estimate_tokens(p) for p in prompt])
                    return prompt_tokens + completion_tokens * len(prompt)
                else:
                    raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
        elif api_endpoint == "embeddings":
            input = request_json["input"]
            if isinstance(input, str):
                return estimate_tokens(input)
            elif isinstance(input, list):
                return sum([estimate_tokens(i) for i in input])
            else:
                raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
        else:
            raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented for Gemini')
    else:
        raise ValueError(f"Unknown api_type: {api_type}")


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1