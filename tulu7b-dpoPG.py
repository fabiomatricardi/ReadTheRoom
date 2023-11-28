import gradio as gr
import os
from ctransformers import AutoModelForCausalLM, AutoConfig, Config #import for GGML models
import datetime

#MODEL SETTINGS also for DISPLAY
modelfile = "model/tulu-2-dpo-7b.Q4_K_M.gguf"
repetitionpenalty = 1.15
contextlength=4096
logfile = 'tulu7bDPO-GGUF-promptsPlayground.txt'
print("loading model...")
stt = datetime.datetime.now()
conf = AutoConfig(Config(temperature=0.3, repetition_penalty=repetitionpenalty, batch_size=64,
                max_new_tokens=2048, context_length=contextlength))
llm = AutoModelForCausalLM.from_pretrained(modelfile,
                                        model_type="llama", config = conf)
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")

def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

"""
gr.themes.Base()
gr.themes.Default()
gr.themes.Glass()
gr.themes.Monochrome()
gr.themes.Soft()
"""
def combine(a, b, c, d):
    import datetime
    temperature = c
    max_new_tokens = d
    prompt = "<|user|>\n" + b + "\n<|assistant|>"
    start = datetime.datetime.now()
    output = llm(prompt, 
                 temperature = temperature, 
                 repetition_penalty = 1.15, 
                 max_new_tokens=max_new_tokens)
    delta = datetime.datetime.now() - start
    #logger = f"""PROMPT: \n{prompt}\nClaude2Alpaca-7b: \n{output}\nGenerated in {delta}\n\n---\n\n"""
    #writehistory(logger)
    generation = f"""{output} """
    prompt_tokens = len(llm.tokenize(prompt))
    answer_tokens = len(llm.tokenize(output))
    total_tokens = prompt_tokens + answer_tokens
    timestamp = datetime.datetime.now()
    logger = f"""time: {timestamp}\n Temp: {temperature} - MaxNewTokens: {max_new_tokens} - RepPenalty: 1.5 \nPROMPT: \n{prompt}\nTulu2-7b-dpo: {output}\nGenerated in {delta}\nPromptTokens: {prompt_tokens}   Output Tokens: {answer_tokens}  Total Tokens: {total_tokens}\n\n---\n\n"""
    writehistory(logger)
    return generation, delta, prompt_tokens, answer_tokens, total_tokens    



# MAIN GRADIO INTERFACE
with gr.Blocks(theme=gr.themes.Soft()) as demo:   #theme=gr.themes.Glass()
    #TITLE SECTION
    with gr.Row(variant='compact'):
            gr.HTML("<center>"
            + "<p>Prompt Engineering Playground - Test your favourite LLM for advanced inferences</p>"
            + "<h1>🐫 Tulu2-7b-DPO 4K context window</h1></center>")                    

    # MODEL PARAMETERS INFO SECTION
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            - **Prompt Template**: Tulu 🐫
            - **Repetition Penalty**: {repetitionpenalty}\n
            """)                            
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            - **Context Lenght**: {contextlength} tokens
            - **LLM Engine**: CTransformers
            """)        
        with gr.Column(scale=2):
            gr.Markdown(
            f"""
            - **Model**: {modelfile}
            - **Log File**: {logfile}
            """)
    # PLAYGROUND INTERFACE SECTION
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            ### Tunning Parameters""")
            temp = gr.Slider(label="Temperature",minimum=0.0, maximum=1.0, step=0.01, value=0.1)
            max_len = gr.Slider(label="Maximum output lenght", minimum=10,maximum=3100,step=2, value=1024)
            gr.Markdown(
            """
            Fill the System Prompt and User Prompt
            And then click the Button below
            """)
            btn = gr.Button(value="🐫 Generate", variant='primary')
            gentime = gr.Textbox(value="", label="🕑 Generation Time:")
            prompttokens = gr.Textbox(value="", label="Prompt Tkn")
            outputokens = gr.Textbox(value="", label="Output Tkn")
            totaltokens = gr.Textbox(value="", label="Total Tokens:")            
        with gr.Column(scale=4):
            txt = gr.Textbox(label="System Prompt", value="This model soes not have a system prompt...",lines=3, interactive=False)
            txt_2 = gr.Textbox(label="User Prompt", lines=5)
            txt_3 = gr.Textbox(value="", label="Output", lines = 10, show_copy_button=True)
            btn.click(combine, inputs=[txt, txt_2,temp,max_len], outputs=[txt_3,gentime,prompttokens,outputokens,totaltokens])

if __name__ == "__main__":
    demo.launch(inbrowser=True)