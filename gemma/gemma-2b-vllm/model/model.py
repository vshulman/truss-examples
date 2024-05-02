import json
import subprocess
import uuid
from typing import Any

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

import os


class Model:
    def __init__(self, **kwargs) -> None:
        self.engine_args = kwargs["config"]["model_metadata"]["engine_args"]
        self.prompt_format = kwargs["config"]["model_metadata"]["prompt_format"]
        self._secrets = kwargs["secrets"]
        
        if "hf_access_token" in self._secrets._base_secrets.keys():
            # Set the environment variable
            os.environ["HF_TOKEN"] = self._secrets["hf_access_token"]
            
    # Optional: to make this tensor parallel        
    if False:
        command = "ray start --head"
        subprocess.check_output(command, shell=True, text=True)
        # add tensor_parallel_size: 2 under engine args

    def load(self) -> None:
        self.llm_engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(**self.engine_args)
        )
        
    def _serialize_logprobs(self, logprobs):
        """
        Serialize logprobs which are in the form of dictionaries with token indices as keys
        and Logprob namedtuples as values into a JSON serializable list of dictionaries.
        """
        serialized_dict = {}
        for token_index, logprob in logprobs.items(): 
            serialized_dict[token_index] = {
                "logprob": logprob.logprob,
                "rank": logprob.rank,
                "decoded_token": logprob.decoded_token
            }
        return serialized_dict


    async def predict(self, request: dict) -> Any:
        prompt = request.pop("prompt")
        stream = request.pop("stream", True)
        formatted_prompt = self.prompt_format.replace("{prompt}", prompt)

        generate_args = {
            "n": 1,
            "best_of": 1,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "frequency_penalty": 1.0,
            "presence_penalty": 1.0,
            "use_beam_search": False,
        }
        generate_args.update(request)

        sampling_params = SamplingParams(**generate_args)
        idx = str(uuid.uuid4().hex)
        vllm_generator = self.llm_engine.generate(
            formatted_prompt, sampling_params, idx
        )
        
        is_logprobs = generate_args.get('logprobs', False)
            
        if is_logprobs and stream:
            raise ValueError("Logprobs and stream cannot be used together")

        async def generator():
            full_text = ""
            async for output in vllm_generator:
                text = output.outputs[0].text
                if is_logprobs:
                    logprobs = output.outputs[0].logprobs[0]
                delta = text[len(full_text) :]
                full_text = text
                if is_logprobs:
                    yield {"delta": delta, "logprobs": self._serialize_logprobs(logprobs)}
                else:
                    yield delta

        if stream:
            return generator()
        else:
            full_text = ""
            full_logprobs = []
            async for response in generator():
                if is_logprobs:
                    full_text += response['delta']
                    full_logprobs.append(response["logprobs"])
                else:
                    full_text += response
                
            if is_logprobs:
                return {"text": full_text, "logprobs": full_logprobs}
            else:
                return full_text
            
   