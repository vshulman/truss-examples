import json
import subprocess
import time
from typing import Any, Dict, List
import httpx  # Changed from aiohttp to httpx


class Model:
    MAX_FAILED_SECONDS = 600  # 10 minutes; the reason this would take this long is mostly if we download a large model

    def __init__(self, data_dir, config, secrets):
        self._secrets = secrets
        self._config = config
        
        # TODO: uncomment for multi-GPU support
        # command = "ray start --head"
        # subprocess.check_output(command, shell=True, text=True)

    def load(self):
        # start the vLLM OpenAI Server
        # TODO: how do we know it's ready?
        # TODO: For ultravox, the first request should happen during load to warm the model
        print(self._config)
        self._vllm_config = self._config["model_metadata"]["arguments"]
        
        command = ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        for key, value in self._vllm_config.items():
            command.append(f"--{key.replace('_', '-')}")
            command.append(str(value))
        
        subprocess.Popen(command)
        
        if 'port' in self._vllm_config:
            self._vllm_port = self._vllm_config['port']
        else:
            self._vllm_port = 8000

    async def predict(self, model_input):
        print(model_input)
        # if model is missing from model_input, use the model from the config
        # Uncommented since there are other issues preventing the bridge from working
        # if 'model' not in model_input and 'model' in self._vllm_config:
        #     print(f"model_input missing model due to Baseten bridge, using {self._vllm_config['model']}")
        #     model_input['model'] = self._vllm_config['model']

        stream = model_input.get('stream', False)
        if stream:
            async def generator():
                async with httpx.AsyncClient(timeout=None) as client:   
                    async with client.stream(
                        "POST",
                        f"http://localhost:{self._vllm_port}/v1/chat/completions",
                    json=model_input
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                yield chunk
                        
            return generator()
        else:
            async with httpx.AsyncClient(timeout=None) as client:   
                response = await client.post(
                    f"http://localhost:{self._vllm_port}/v1/chat/completions",
                        json=model_input
                    )
                return response.json()

