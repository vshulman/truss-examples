import json
import subprocess
import time
from typing import Any, Dict, List
import httpx  # Changed from aiohttp to httpx


class Model:
    MAX_FAILED_SECONDS = 600  # 10 minutes; the reason this would take this long is mostly if we download a large model

    def __init__(self, data_dir, config, secrets):
        # TODO: whats teh right way to accept args?
        self._secrets = secrets
        self._config = config
        
        # command = "ray start --head"
        # subprocess.check_output(command, shell=True, text=True)

    def load(self):
        # start the vLLM OpenAI Server
        # TODO: how do we know it's ready?
        print(self._config)
        vllm_config = self._config["model_metadata"]["arguments"]
        
        command = ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        for key, value in vllm_config.items():
            command.append(f"--{key.replace('_', '-')}")
            command.append(str(value))
        
        subprocess.Popen(command)
        
        if 'port' in self._config['model_metadata']['arguments']:
            self._vllm_port = vllm_config['model_metadata']['arguments']['port']
        else:
            self._vllm_port = 8000

    async def predict(self, model_input):
        print(model_input)
        # if model is missing from model_input, use the model from the config
        if 'model' not in model_input:
            print(f"model_input missing model due to Baseten bridge, using {self._config['model_metadata']['name']}")
            model_input['model'] = self._config['model_metadata']['name']

        stream = model_input.get('stream', False)
        # https://github.com/vllm-project/vllm/blob/665c48963be11b2e5cb7209cd25f884129e5c284/examples/api_client.py#L26
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

