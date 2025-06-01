import torch
import transformers
from threading import Thread

pipe: transformers.pipelines.Pipeline | None = None

type Streamer = transformers.TextIteratorStreamer | transformers.TextStreamer | transformers.generation.streamers.BaseStreamer
type ModelOutput = str | dict[str, list[dict[str, str]]]

def setup(model: str | None = "meta-llama/Llama-3.2-3B-Instruct", token = None):
    global pipe
    pipe = transformers.pipeline(
        "text-generation",
        model=model or "meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token = token,
        framework='pt'
    )

def generate(prompt: str, stream: Streamer, **kwargs) -> tuple[Thread, Streamer] | ModelOutput:
    global pipe
    if not pipe:
        setup()
    if stream:
        args = kwargs
        args['streamer'] = stream
        t = Thread(target=pipe, args=(prompt,), kwargs=args)
        t.start()
        return (t, stream)
    else:
        return pipe(prompt, **kwargs)
