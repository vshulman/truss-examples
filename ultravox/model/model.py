from threading import Thread
from typing import Dict, Optional
import time

from ultravox.data import datasets
from ultravox.inference import ultravox_infer
from ultravox.inference import base

# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     GenerationConfig,
#     TextIteratorStreamer,
# )

MAX_LENGTH = 512
TEMPERATURE = 1.0
DEFAULT_STREAM = True


class UltravoxInferenceArgs:
    def __init__(
        self,
        model_path: str,
        audio_processor_id: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        device: Optional[str] = None,
        data_type: Optional[str] = None,
    ):
        self.model_path = model_path
        self.audio_processor_id = audio_processor_id
        self.tokenizer_id = tokenizer_id
        self.device = device
        self.data_type = data_type


class Model:
    def __init__(self, **kwargs):
        self.audio_processor_id = None
        self.data_type = None
        self._secrets = kwargs["secrets"]
        self.hf_access_token = self._secrets["hf_access_token"] # useful later if we have a private / gated hf repo

    def load(self):
        hf_path = self.config['model_cache'].get('repo_id')
        model_metadata = self.config.get('model_metadata', {})

        # Filter out only the relevant keys for UltravoxInferenceArgs
        relevant_keys = ['model_path', 'audio_processor_id', 'tokenizer_id', 'device', 'data_type']
        filtered_metadata = {k: v for k, v in model_metadata.items() if k in relevant_keys}

        inference_args = UltravoxInferenceArgs(
            model_path=hf_path,
            **filtered_metadata
        )

        inference = ultravox_infer.UltravoxInference(**vars(inference_args))

    # TODO: update from_json in datasets to handle this?
    def json_to_sample(self, message):
        content = message["content"]
        text_parts = [part["text"] for part in content if part["type"] == "text"]
        audio_url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
        # clone message
        mock_payload = {}
        mock_payload['audio'] = audio_url.split(',', 1)[1]
        mock_payload["messages"] = [message]
        # datasets.VoiceSample.from_prompt_and_raw
        sample = datasets.VoiceSample.from_json(mock_payload)
        # sample = datasets.VoiceSample(messages=[{"role": "user", "content": "".join(text_parts)}], audio=audio_data)
        return sample

    def stream_inference(self, sample, max_tokens, temperature):
        stream = self.inference.infer_stream(sample, max_tokens=max_tokens, temperature=temperature)
        first_token_time = None
        stats = None
        for msg in stream:
            if isinstance(msg, base.InferenceChunk):
                if first_token_time is None:
                    first_token_time = time.time()
                yield msg
            elif isinstance(msg, base.InferenceStats):
                stats = msg
        if first_token_time is None or stats is None:
            raise ValueError("No tokens received")
        yield {"stats": stats}

    def predict(self, request: Dict):
        messages = request.pop("messages")
        stream_results = request.pop("stream", DEFAULT_STREAM)
        max_tokens = request.pop("max_tokens", MAX_LENGTH)
        temperature = request.pop("temperature", TEMPERATURE)

        # Convert messages back into VoiceSample
        sample = self.json_to_sample(messages[0])

        if stream_results:
            return self.stream_inference(sample, max_tokens, temperature)
        else:
            result = self.inference.infer(sample, max_tokens=max_tokens, temperature=temperature)
            text = result.text
            stats = result.stats
            return {"output": text, "stats": stats}


# chunks: {"id":"cmpl-e9c0822446d74ab78310263b61a92cfe","object":"chat.completion.chunk","created":1719167921,"model":"fixie-ai/ultravox-v0.2","choices":[{"index":0,"delta":{"content":" regions"},"logprobs":null,"finish_reason":null}]}
