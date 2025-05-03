# Taken and adapted from https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py

# imports
import sys
import aiohttp  # for making API calls concurrently
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
from medvqa.utils.files_utils import load_jsonl, load_pickle, save_jsonl, save_pickle  # for storing API inputs, outputs, and metadata

__all__ = [
    "process_api_requests_from_file",
    "GPT_IS_ACTING_WEIRD_REGEX",
    "run_common_boilerplate_for_api_requests",
]

_ALLOWED_GPT_CHAT_MODELS = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-1106-preview",
    "gpt-4o",
    "gpt-4o-mini",
)

GPT_IS_ACTING_WEIRD_REGEX = re.compile(r"\b(I'm sorry|Sorry|Could you|Can you|I apologize|Sure|I'd be happy|do you need)\b")

def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str = None,
    request_url: str = "https://api.openai.com/v1/embeddings",
    api_key: str = os.getenv("OPENAI_API_KEY"),
    max_requests_per_minute: float = 3_000 * 0.5,
    max_tokens_per_minute: float = 250_000 * 0.5,
    token_encoding_name: str = "cl100k_base",
    max_attempts: int = 5,
    logging_level: int = logging.INFO,
    log_info_every_n_requests: int = 100,
):
    assert os.path.exists(requests_filepath), f"API requests file {requests_filepath} does not exist"
    assert api_key is not None, "API key must be provided"
    assert request_url is not None, "Request URL must be provided"
    assert max_requests_per_minute is not None, "Max requests per minute must be provided"
    assert max_tokens_per_minute is not None, "Max tokens per minute must be provided"
    assert token_encoding_name is not None, "Token encoding name must be provided"
    assert max_attempts is not None, "Max attempts must be provided"

    if save_filepath is None:
        save_filepath = requests_filepath.replace(".jsonl", "_results.jsonl")

    return asyncio.run(
        __process_api_requests_from_file(
            requests_filepath=requests_filepath,
            save_filepath=save_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=max_attempts,
            logging_level=logging_level,
            log_info_every_n_requests=log_info_every_n_requests,
        )
    )

def _generate_request(system_instructions, query, model_name, max_tokens,
                     temperature=0, frequency_penalty=0, presence_penalty=0):
    assert len(system_instructions) > 0
    assert len(query) > 0
    assert model_name in _ALLOWED_GPT_CHAT_MODELS, f"Unknown model name: {model_name}"
    return {
        "model": model_name,
        "messages": [{
            "role": "system",
            "content": system_instructions,
        }, {
            "role": "user",
            "content": query,
        }],
        "temperature": temperature, # 0.0 = (almost) deterministic, 2.0 = max entropy
        "frequency_penalty": frequency_penalty, # 0.0 = no penalty, 2.0 = max penalty
        "presence_penalty": presence_penalty, # 0.0 = no penalty, 2.0 = max penalty
        "max_tokens": max_tokens,
        "metadata": {
            "query": query,
        },
    }

# Based on https://platform.openai.com/docs/guides/batch/getting-started
def _generate_batch_request(custom_id, model_name, system_instructions, query, max_tokens,
                            temperature=0.0, frequency_penalty=0.0, presence_penalty=0.0):
    assert len(system_instructions) > 0
    assert len(query) > 0
    assert model_name in _ALLOWED_GPT_CHAT_MODELS, f"Unknown model name: {model_name}"
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
        api_responses_filepath,
        texts, system_instructions, api_key_name, openai_model_name, openai_request_url,
        max_tokens_per_request, max_requests_per_minute, max_tokens_per_minute,
        temperature, frequency_penalty, presence_penalty,
        logger, logging_level, parse_openai_output, tmp_dir, save_filepath,
        delete_api_requests_and_responses=True,
        use_batch_api=False,
        batch_description=None,
        batch_input_file_id=None,
        ):
    """Runs common boilerplate for API requests."""

    if api_responses_filepath is None:

        if use_batch_api:
            
            if batch_input_file_id is not None:
                batch_object_filepath = os.path.join(tmp_dir, "openai", "batch_api", f"{batch_input_file_id}.pkl")
                assert os.path.exists(batch_object_filepath), f"Batch object file {batch_object_filepath} does not exist"
                logger.info(f"Checking status of batch with input file ID {batch_input_file_id}")
                
                batch_object = load_pickle(batch_object_filepath)
                logger.info(f"Batch object: {batch_object}")
                
                # We need to check the status of an existing batch already created through the API
                from openai import OpenAI
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
                            parsed_output = parse_openai_output(text)
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
                jobs = []
                for i, text in enumerate(texts):
                    jobs.append(_generate_batch_request(
                        custom_id=str(i),
                        model_name=openai_model_name,
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
                save_jsonl(jobs, batch_api_requests_filepath)

                from openai import OpenAI
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
            # Prepare API requests
            jobs = []
            for text in texts:
                jobs.append(_generate_request(
                    system_instructions=system_instructions,
                    query=text,
                    model_name=openai_model_name,
                    max_tokens=max_tokens_per_request,
                    temperature=temperature,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                ))
                assert 'metadata' in jobs[-1]
            
            timestamp = get_timestamp()
            api_requests_filepath = os.path.join(tmp_dir, "openai", f"api_requests_{timestamp}.jsonl")
            api_responses_filepath = os.path.join(tmp_dir, "openai", f"api_responses_{timestamp}.jsonl")
            logger.info(f"Saving API requests to {api_requests_filepath}")
            logger.info(f"Saving API responses to {api_responses_filepath}")
            save_jsonl(jobs, api_requests_filepath)

            # Send API requests
            process_api_requests_from_file(
                requests_filepath=api_requests_filepath,
                save_filepath=api_responses_filepath,
                request_url=openai_request_url,
                api_key=os.getenv(api_key_name),
                max_requests_per_minute=max_requests_per_minute,
                max_tokens_per_minute=max_tokens_per_minute,
                token_encoding_name=tiktoken.encoding_for_model(openai_model_name).name,
                max_attempts=5,
                logging_level=logging_level,
                log_info_every_n_requests=50,
            )

    else:
        assert os.path.exists(api_responses_filepath), f"API responses file {api_responses_filepath} does not exist"
        logger.info(f"API responses already exist at {api_responses_filepath}. Skipping API requests.")
        dir_path = os.path.dirname(api_responses_filepath)
        filename = os.path.basename(api_responses_filepath)
        api_requests_filepath = os.path.join(dir_path, filename.replace("responses", "requests"))
        assert os.path.exists(api_requests_filepath), f"API requests file {api_requests_filepath} does not exist"
        jobs = None

    # Load and postprocess API responses
    logger.info(f"Loading API responses from {api_responses_filepath}")
    api_responses = load_jsonl(api_responses_filepath)
    if jobs is not None:
        assert len(api_responses) == len(jobs)

    postprocessed_responses = []
    for i in range(len(api_responses)):
        api_response = api_responses[i]
        assert len(api_response) == 3 # request, response, and metadata
        metadata = api_response[2]
        try:
            text = api_response[1]['choices'][0]['message']['content']
            parsed_output = parse_openai_output(text)
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
        logger.info(f"""Succesfully processed {n_processed} of {n_total} API responses.
                    {n_total - n_processed} of {n_total} API responses could not be processed.
                    Saving processed texts to {save_filepath}""")
        save_jsonl(postprocessed_responses, save_filepath, append=True)

async def __process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
    log_info_every_n_requests: int = 100,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}

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
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")

        while True:
            # get next request (if one is not already waiting for capacity)
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                    logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
                elif file_not_finished:
                    try:
                        # get new request
                        request_json = json.loads(next(requests))
                        next_request = APIRequest(
                            task_id=next(task_id_generator),
                            request_json=request_json,
                            token_consumption=num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name),
                            attempts_left=max_attempts,
                            metadata=request_json.pop("metadata", None)
                        )
                        status_tracker.num_tasks_started += 1
                        status_tracker.num_tasks_in_progress += 1
                        logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                    except StopIteration:
                        # if file runs out, set flag to stop reading it
                        logging.debug("Read file exhausted")
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
                next_request_tokens = next_request.token_consumption
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
                            request_url=request_url,
                            request_header=request_header,
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
                logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

        # after finishing, log final status
        logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
        if status_tracker.num_tasks_failed > 0:
            logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}.")
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")


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

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
        log_info: bool = True
    ):
        """Calls the OpenAI API and saves results."""
        if log_info:
            logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")

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


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
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


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1