# vLLM OpenAI Compatible Server (ChatCompletions only)
The Truss demonstrates how to start vLLM's OpenAI compatible server. 
The Truss is primarily used to start the server, and then route requests to it.

## Passing startup arguments to the server
In the config, create a top-level server-arguments key. Any key-values under it will be passed to the server at startup.

## Base Image
The base image is the same as the one used in the vLLM project.

# TODO
- Add support for distributed serving with Ray https://docs.vllm.ai/en/latest/serving/distributed_serving.html
- Consider caching the model
- Consider defaulting to uvicorn-log-level log level warning
- Ensure we have verbose logging on failure