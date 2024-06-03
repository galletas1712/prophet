import json
import logging
import time

import numpy as np

from entrypoints.api import Request
from models.llama3.tokenizer import Tokenizer, LlamaFormatter

shareGPTPath = '/home/ubuntu/shareGPT.json'
tokenizer = Tokenizer(
    '/home/ubuntu/model_weights/Meta-Llama-3-8B-Instruct/tokenizer.model')
formatter = LlamaFormatter(tokenizer)


def read_shareGPTJSON():
    f = open(shareGPTPath)
    shareGPTJSON = json.load(f)
    f.close()
    return shareGPTJSON


def shareGPT_to_llama_format(entry):
    if entry['from'] == 'human':
        return {
            'role': 'user',
            'content': entry['value']
        }
    else:
        return {
            'role': 'assistant',
            'content': entry['value']
        }


def preprocess_shareGPT_dialog(dialog, max_tokens):
    dialog = list(map(shareGPT_to_llama_format, dialog))
    dialog_len = formatter.get_max_dialog_len(dialog, max_tokens)
    while dialog_len > 0 and dialog[dialog_len - 1]['role'] == 'assistant':
        dialog_len -= 1
    return dialog[:dialog_len]


def preprocess_shareGPT_dialogs(corpus, max_tokens):
    return filter(
        lambda x: len(x) > 0,
        map(
            lambda convo: preprocess_shareGPT_dialog(
                convo['conversations'], max_tokens),
            corpus
        )
    )


def request_generator(request_queue, num_termination_requests):
    logger = logging.getLogger("shareGPT.request_generator")
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting request generator...")
    shareGPTJSON = read_shareGPTJSON()
    dialogs = preprocess_shareGPT_dialogs(shareGPTJSON, 300)

    num_secs = 180
    arrivals_per_sec = 0.5

    for _ in range(1, num_secs):
        time.sleep(1)
        num_requests = np.random.poisson(arrivals_per_sec)
        logger.debug(f"Current timestep requests: {num_requests}")
        for _ in range(num_requests):
            request = Request(
                next(dialogs),
                np.random.randint(5, 450),
            )
            logger.info(f"START {request.request_id} prompt_len {len(
                formatter.encode_chat_completion(request.prompt))} max_gen_len {request.max_gen_len}")
            request_queue.put(request)

    for _ in range(num_termination_requests):
        request_queue.put(None)
