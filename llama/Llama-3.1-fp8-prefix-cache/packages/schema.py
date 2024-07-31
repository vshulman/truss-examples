from typing import Optional, List
import numpy as np
import tritonclient
import tritonclient.grpc.aio as grpcclient

class ModelInput:
    def __init__(
        self,
        prompt: str,
        request_id: int,
        max_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        beam_width: int = 1,
        bad_words_list: Optional[List[str]] = None,
        stop_words_list: Optional[List[str]] = None,
        end_id: Optional[int] = None,
        pad_id: Optional[int] = None,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.0,
        min_length: int = 0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        random_seed: Optional[int] = None,
        return_log_probs: bool = False,
        return_context_logits: bool = False,
        return_generation_logits: bool = False,
        stream: bool = True,
        decoder_input: Optional[str] = None,
        image_input: Optional[np.ndarray] = None,
        prompt_embedding_table: Optional[np.ndarray] = None,
        prompt_vocab_size: Optional[int] = None,
        embedding_bias_words: Optional[List[str]] = None,
        embedding_bias_weights: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        self.request_id = request_id
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._beam_width = beam_width
        self._bad_words_list = bad_words_list or [""]
        self._stop_words_list = stop_words_list or [""]
        self._end_id = end_id
        self._pad_id = pad_id
        self._length_penalty = length_penalty
        self._repetition_penalty = repetition_penalty
        self._min_length = min_length
        self._presence_penalty = presence_penalty
        self._frequency_penalty = frequency_penalty
        self._random_seed = random_seed
        self._return_log_probs = return_log_probs
        self._return_context_logits = return_context_logits
        self._return_generation_logits = return_generation_logits
        self._stream = stream
        self._decoder_input = decoder_input
        self._image_input = image_input
        self._prompt_embedding_table = prompt_embedding_table
        self._prompt_vocab_size = prompt_vocab_size
        self._embedding_bias_words = embedding_bias_words
        self._embedding_bias_weights = embedding_bias_weights

    def _prepare_grpc_tensor(
        self, name: str, input_data: np.ndarray
    ) -> grpcclient.InferInput:
        tensor = grpcclient.InferInput(
            name,
            input_data.shape,
            tritonclient.utils.np_to_triton_dtype(input_data.dtype),
        )
        tensor.set_data_from_numpy(input_data)
        return tensor

    def to_tensors(self):
        inputs = [
            self._prepare_grpc_tensor("text_input", np.array([[self._prompt]], dtype=object)),
            self._prepare_grpc_tensor("max_tokens", np.array([[self._max_tokens]], dtype=np.int32)),
        ]

        if self._decoder_input is not None:
            inputs.append(self._prepare_grpc_tensor("decoder_text_input", np.array([[self._decoder_input]], dtype=object)))

        if self._image_input is not None:
            inputs.append(self._prepare_grpc_tensor("image_input", self._image_input))

        if self._bad_words_list:
            inputs.append(self._prepare_grpc_tensor("bad_words", np.array([self._bad_words_list], dtype=object)))

        if self._stop_words_list:
            inputs.append(self._prepare_grpc_tensor("stop_words", np.array([self._stop_words_list], dtype=object)))

        if self._end_id is not None:
            inputs.append(self._prepare_grpc_tensor("end_id", np.array([[self._end_id]], dtype=np.int32)))

        if self._pad_id is not None:
            inputs.append(self._prepare_grpc_tensor("pad_id", np.array([[self._pad_id]], dtype=np.int32)))

        inputs.extend([
            self._prepare_grpc_tensor("top_k", np.array([[self._top_k]], dtype=np.int32)),
            self._prepare_grpc_tensor("top_p", np.array([[self._top_p]], dtype=np.float32)),
            self._prepare_grpc_tensor("temperature", np.array([[self._temperature]], dtype=np.float32)),
            self._prepare_grpc_tensor("length_penalty", np.array([[self._length_penalty]], dtype=np.float32)),
            self._prepare_grpc_tensor("repetition_penalty", np.array([[self._repetition_penalty]], dtype=np.float32)),
            self._prepare_grpc_tensor("min_length", np.array([[self._min_length]], dtype=np.int32)),
            self._prepare_grpc_tensor("presence_penalty", np.array([[self._presence_penalty]], dtype=np.float32)),
            self._prepare_grpc_tensor("frequency_penalty", np.array([[self._frequency_penalty]], dtype=np.float32)),
            self._prepare_grpc_tensor("return_log_probs", np.array([[self._return_log_probs]], dtype=bool)),
            self._prepare_grpc_tensor("return_context_logits", np.array([[self._return_context_logits]], dtype=bool)),
            self._prepare_grpc_tensor("return_generation_logits", np.array([[self._return_generation_logits]], dtype=bool)),
            self._prepare_grpc_tensor("beam_width", np.array([[self._beam_width]], dtype=np.int32)),
            self._prepare_grpc_tensor("stream", np.array([[self._stream]], dtype=bool)),
        ])

        if self._random_seed is not None:
            inputs.append(self._prepare_grpc_tensor("random_seed", np.array([[self._random_seed]], dtype=np.uint64)))

        if self._prompt_embedding_table is not None:
            inputs.append(self._prepare_grpc_tensor("prompt_embedding_table", self._prompt_embedding_table))

        if self._prompt_vocab_size is not None:
            inputs.append(self._prepare_grpc_tensor("prompt_vocab_size", np.array([[self._prompt_vocab_size]], dtype=np.int32)))

        if self._embedding_bias_words is not None:
            inputs.append(self._prepare_grpc_tensor("embedding_bias_words", np.array([self._embedding_bias_words], dtype=object)))

        if self._embedding_bias_weights is not None:
            inputs.append(self._prepare_grpc_tensor("embedding_bias_weights", np.array([self._embedding_bias_weights], dtype=np.float32)))

        return inputs