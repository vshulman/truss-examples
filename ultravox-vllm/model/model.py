import subprocess
import time
from typing import Any, Dict, List
import aiohttp


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
        stream = model_input.get('stream', False)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:{self._vllm_port}/v1/chat/completions",
                json=model_input
            ) as response:
                
                async def generator():
                    async for chunk in response.content.iter_chunked(8192):
                        if chunk:
                            json_chunk = await response.json()
                            yield json_chunk

                if stream:
                    return generator()
                else:
                    return await response.json()
