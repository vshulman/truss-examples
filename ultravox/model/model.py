from typing import Dict, Optional
import time

from ultravox.data import datasets
from ultravox.inference import ultravox_infer
from ultravox.inference import base

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
        self._config = kwargs["config"]
        self.hf_access_token = self._secrets["hf_access_token"] # useful later if we have a private / gated hf repo

    def load(self):
        hf_path = self._config['model_cache'][0].get('repo_id')
        model_metadata = self._config.get('model_metadata', {})

        # Filter out only the relevant keys for UltravoxInferenceArgs
        relevant_keys = ['model_path', 'audio_processor_id', 'tokenizer_id', 'device', 'data_type']
        filtered_metadata = {k: v for k, v in model_metadata.items() if k in relevant_keys}

        inference_args = UltravoxInferenceArgs(
            model_path=hf_path,
            **filtered_metadata
        )

        self.inference = ultravox_infer.UltravoxInference(**vars(inference_args))

    # TODO: 1. Add this to datasets, 2. how to handle multiple messages and audio files?
    # Converts from oai json to standard json and then to voice sample
    def oai_json_to_sample(self, message):
        content = message["content"]
        role = message["role"]
        
        # Extract and delete the image_url part
        audio_url = None
        new_content = []
        for part in content:
            if part["type"] == "image_url" and audio_url is None:
                audio_url = part["image_url"]["url"]
            else:
                new_content.append(part)
        
        # Prepare modified payload
        payload = {}
        if audio_url:
            payload['audio'] = audio_url.split(',', 1)[1]
        
        # Copy the original message and update its content
        payload["messages"] = [{"role": role, "content": new_content}]
        
        # Convert to VoiceSample
        sample = datasets.VoiceSample.from_json(payload)
        
        return sample

    def stream_inference(self, sample, max_tokens, temperature):
        stream = self.inference.infer_stream(sample, max_tokens=max_tokens, temperature=temperature)
        first_token_time = None
        stats = None
        for msg in stream:
            if isinstance(msg, base.InferenceChunk):
                if first_token_time is None:
                    first_token_time = time.time()
                yield msg.text
            elif isinstance(msg, base.InferenceStats):
                stats = msg
        if first_token_time is None or stats is None:
            raise ValueError("No tokens received")
        # yield {"stats": stats}

    def predict(self, request: Dict):
        messages = request.pop("messages")
        stream_results = request.pop("stream", DEFAULT_STREAM)
        max_tokens = request.pop("max_tokens", MAX_LENGTH)
        temperature = request.pop("temperature", TEMPERATURE)

        # TODO: How to handle multiple messages?
        sample = self.oai_json_to_sample(messages[0])

        if stream_results:
            return self.stream_inference(sample, max_tokens, temperature)
        else:
            result = self.inference.infer(sample, max_tokens=max_tokens, temperature=temperature)
            text = result.text
            # stats = result.stats
            return {"output": text} #, "stats": stats}


# chunks: {"id":"cmpl-e9c0822446d74ab78310263b61a92cfe","object":"chat.completion.chunk","created":1719167921,"model":"fixie-ai/ultravox-v0.2","choices":[{"index":0,"delta":{"content":" regions"},"logprobs":null,"finish_reason":null}]}
