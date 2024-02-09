# ReadTheRoom
Repo of the code from the Medium article Read the room… learn when it is time to stop

updated on 20240209
## Stop! Don't read this until you get your LLM Under Control
#### Learn crucial "stop words" to avoid information overload and unlock focused conversations with your Large Language Model.

---

### Instructions
- create a virtual environment
- activate it
- install the following packages
```
pip install ctransformers>=0.2.24
pip install gradio
pip install llama-cpp-python
```

### Model weights
Download from huggingface/TheBloke your quantized model files
and put them in the subdirectory `model`
- https://huggingface.co/TheBloke/claude2-alpaca-7B-GGUF
- https://huggingface.co/TheBloke/Orca-2-7B-GGUF
- https://huggingface.co/TheBloke/tulu-2-dpo-7B-GGUF

Changes in the code if using llama-cpp-python

```
import gradio as gr
import os
import datetime
from llama_cpp import Llama

#MODEL SETTINGS also for DISPLAY
modelfile = "model/orca-2-7b.Q4_K_M.gguf"
repetitionpenalty = 1.15
contextlength=4096
logfile = 'Orca2-7b-GGUF-promptsPlayground.txt'
print("loading model...")
  llm = Llama(
    model_path=modelfile,  # Download the model file first
    n_ctx=contextlength,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=2,            # The number of CPU threads to use, tailor to your system and the resulting performance
  )

```


Call the inference

```
prompt = "<|im_start|>system\n"+ a 
         + "<|im_end|>\n<|im_start|>user\n" + b 
         + "<|im_end|>\n<|im_start|>assistant"
output = llm(prompt, 
          temperature = temperature, 
          repetition_penalty = 1.15, 
          max_new_tokens=max_new_tokens,
          stop = ['USER'],
          echo=False)    #this is the only difference

```

