import os
from itertools import count

from schema import ModelInput
from transformers import AutoTokenizer
from triton_client import TritonClient, TritonServer
from truss.constants import OPENAI_COMPATIBLE_TAG

from constants import (
    GRPC_SERVICE_PORT,
    HF_AUTH_KEY_CONSTANT,
    HTTP_SERVICE_PORT,
    TOKENIZER_KEY_CONSTANT,
)

DEFAULT_MAX_TOKENS = 500
DEFAULT_MAX_NEW_TOKENS = 500
REQUIRED_SERVING_KEYS = ["engine_repository", "tokenizer_repository", "tensor_parallel_count"]

class Model:
    def __init__(self, data_dir, config, secrets, lazy_data_resolver):
        self._data_dir = data_dir
        self._config = config
        self._secrets = secrets
        self._request_id_counter = count(start=1)
        self._lazy_data_resolver = lazy_data_resolver
        self.triton_client = None
        self.triton_server = None
        self.tokenizer = None
        self.uses_openai_api = None

    def load(self):
        model_metadata = self._config.get("model_metadata", {})
        if not all(key in model_metadata for key in REQUIRED_SERVING_KEYS):
            raise ValueError("Missing required keys in model_metadata to serve with TRT-LLM")
        self.uses_openai_api = OPENAI_COMPATIBLE_TAG in model_metadata.get("tags", [])

        hf_access_token = None
        if "hf_access_token" in self._secrets._base_secrets.keys():
            hf_access_token = self._secrets["hf_access_token"]

        tokenizer_repository = model_metadata["tokenizer_repository"]
        world_size = model_metadata["tensor_parallel_count"] * model_metadata["pipeline_parallel_count"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_repository, token=hf_access_token
        )
        self.eos_token_id = self.tokenizer.eos_token_id

        # Set up Triton Server
        env = {}
        if hf_access_token:
            env[HF_AUTH_KEY_CONSTANT] = hf_access_token
        env[TOKENIZER_KEY_CONSTANT] = tokenizer_repository

        self.triton_server = TritonServer(
            grpc_port=GRPC_SERVICE_PORT,
            http_port=HTTP_SERVICE_PORT,
        )
        self.triton_server.create_model_repository(
            truss_data_dir=self._data_dir,
            engine_repository_path=model_metadata["engine_repository"],
            huggingface_auth_token=hf_access_token,
        )
        self.triton_server.start(
            world_size=world_size,
            env=env,
        )
        self.triton_client = TritonClient(
            grpc_service_port=GRPC_SERVICE_PORT,
        )

    async def predict(self, model_input):
        if "messages" not in model_input and "prompt" not in model_input:
            raise ValueError("Prompt or messages must be provided")

        model_input.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
        model_input.setdefault("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        model_input["request_id"] = str(os.getpid()) + str(
            next(self._request_id_counter)
        )
        model_input["eos_token_id"] = self.eos_token_id

        if "messages" in model_input:
            messages = model_input.pop("messages")
            if self.uses_openai_api and "prompt" not in model_input:
                model_input["prompt"] = self.tokenizer.apply_chat_template(
                    messages, tokenize=False
                )

        self.triton_client.start_grpc_stream()
        model_input = ModelInput(**model_input)
        result_iterator = self.triton_client.infer(model_input)

        async def generate():
            async for result in result_iterator:
                yield result

        async def build_response():
            full_text = ""
            async for delta in result_iterator:
                full_text += delta
            return full_text

        if model_input._stream:
            return generate()
        else:
            text = await build_response()
            if self.uses_openai_api:
                return text
            else:
                return {"text": text}