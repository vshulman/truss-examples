# vLLM OpenAI Compatible Server (ChatCompletions only)
The Truss demonstrates how to start vLLM's OpenAI compatible server. 
The Truss is primarily used to start the server, and then route requests to it.

## Passing startup arguments to the server
In the config, create a top-level server-arguments key. Any key-values under it will be passed to the server at startup.

## Base Image
The base image is the same as the one used in the vLLM project. Simply ensure:
* Python is available

# TODO
- Enable arbitrary model support
- Enable arbitrary server start arguments
- Support streaming
- Wait the right amount of time for the server to start
- Ensure python is somewhere
- How to handle concurrency?
- How to handle multiple GPUs? https://docs.vllm.ai/en/latest/serving/distributed_serving.html
- Consider caching the model
- Figure out how to handle logs properly
- Ensure we have verbose logging
```
root@69f03e14a9f4:/vllm-workspace# curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
     "model": "fixie-ai/ultravox-v0.2",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

response = requests.post( "http://localhost:8000/v1/chat/completions", json=data ) # data is dict