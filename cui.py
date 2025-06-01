from rich.console import Console
from rich.markdown import Markdown

c = Console()
c.rule('QT | Chat')

c.print("\t\t\t\t\t    Welcome to [bold]QuickThink Chat[/]!")
c.print("[dim]Loading libraries...[/dim]")

import transformers
import torch
import use
import time
import os
import threading
import sys
import keyengine
import colorama
import json
import psutil

def flush_stdin():
    """
    Flush any unread input from sys.stdin so that the leftover ENTER
    doesn’t immediately satisfy the next input() call.
    """
    try:
        # Unix‐style flush
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except (ImportError, OSError):
        # Windows‐style flush
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()

def setup():
    os.system('cls || clear')
    c.rule('QT | Chat')
    c.print("\t\t\t\t\t    Welcome to [bold blue]QuickThink Chat[/]!")
    c.print("\t\t\t\t\t                  [dim]Setup[/]")
    c.print("Please follow the instructions on-screen, pressing [green]ENTER[/] after each completed instruction.")

    time.sleep(0.5)
    c.print(Markdown("1. Please create a [HuggingFace account](https://huggingface.co/join)."), markup=True, end='')
    c.input()
    c.print(Markdown(
        "2. Create a read-only access token [here](https://huggingface.co/settings/tokens/new?tokenType=read)."),
            markup=True, end='')
    c.input()
    TOKEN = c.input("3. Paste the access token here [dim](input will be hid for security reasons)[/]: ", password=True,
                    markup=True).strip()

    if not TOKEN or not TOKEN.startswith("hf_") or len(TOKEN) < 6:
        c.print("[bold red]Invalid token. If you believe this is an error, try creating a new token to use.")
        sys.exit(1)

    c.print(
        "\n\n4. You will soon be prompted to chose which model to use. You will not be able to change this later.\nKeep in mind that the smaller and faster a model is, the higher the chance of it giving misinformation.")
    options = [
        'Choose a model. (Ranked from best to worst)',
        f'{colorama.Style.DIM}{colorama.Fore.RED}1. Llama 4 Scout (109B)     [NOT RECOMMENDED ON CONSUMER-GRADE HARDWARE]{colorama.Style.RESET_ALL}',
        f'{colorama.Style.DIM}{colorama.Fore.RED}2. Llama 3.1 (70B)        [NOT RECOMMENDED ON CONSUMER-GRADE HARDWARE]{colorama.Style.RESET_ALL}',
        '3. Deepseek V2 Lite (16B)     [SLOW]'
        '4. Llama 3.1 (8B)        [SLOW]',
        f'{colorama.Fore.GREEN}5. Llama 3.2 (3B)         [RECOMMENDED]{colorama.Style.RESET_ALL}',
        '6. Llama 3.2 (1B)           [FAST | RECOMMENDED FOR CREATIVE TASKS]',
        f'{colorama.Style.DIM}{colorama.Fore.RED}7. DialoGPT Medium (354M)    [SLOW | NOT ADVISED]{colorama.Style.RESET_ALL}'
    ]
    m = {
        1: 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        2: 'meta-llama/Llama-3.1-70B-Instruct',
        3: 'deepseek-ai/DeepSeek-V2-Lite-Chat',
        4: 'meta-llama/Llama-3.1-8B-Instruct',
        5: 'meta-llama/Llama-3.2-3B-Instruct',
        6: 'meta-llama/Llama-3.2-1B-Instruct',
        7: 'microsoft/DialoGPT-medium'
    }
    l = {
        1: 'https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct',
        2: 'https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct',
        3: 'https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat',
        4: 'https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct',
        5: 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct',
        6: 'https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct',
        7: 'https://huggingface.co/microsoft/DialoGPT-medium'
    }
    time.sleep(2)
    while True:
        try:
            choice = keyengine.menu(options)
            if choice == 0:
                continue
            break
        except Exception as _:
            pass

    os.system('cls || clear')

    flush_stdin()  # Because of Python's shitty buffering system for key inputs

    if choice != 6 and choice != 3:
        c.print(f"You have chosen {m[choice]}. Please wait for new instructions.")
        time.sleep(0.5)
        c.print(Markdown(
            f"5. Please go to the [model page]({l[choice]}) of your chosen model and request access. You should be granted access in <30 minutes (depending on traffic)."),
            end='')
        c.input()
    data = {
        'model': m[choice],
        'mc': choice,
        'token': TOKEN
    }
    with open('config.json', 'w') as f:
        json.dump(data, f, indent=4)

    c.print("Configuration finished.")
    time.sleep(0.2)

if not os.path.exists('config.json'):
    setup()

def get_config() -> dict[str, str]:
    with open('config.json', 'r') as f:
        assert not f.closed, "Failed to open file. If you're seeing this, please open an issue on GitHub (https://github.com/cornusandu/QuickThinkChat)"  # Just to be safe
        assert f.readable(), "Uhm this shouldn't have happened (CODE 1)\nPlease open an issue on GItHub (https://github.com/cornusandu/QuickThinkChat)."  # Shouldn't happen, but if it does, program runtime is fucked
        return json.load(f)

os.system('cls || clear')
c.rule("QT Chat | Installation")
c.print("Please wait while we load required models. If this is the first time you're running the script, we will have to download said models. Please wait, this may take a while. (est. time 5-10 mins)")
time.sleep(0.4)
model = get_config()['model']
TOKEN = get_config()['token']
use.setup(model, token = TOKEN)
time.sleep(0.2)
os.system('cls || clear')

c.rule("QuickThink Chat")
c.print("Talk with your chosen AI chatbot, completely offline!")

proc = max(1, psutil.cpu_count(logical=True) - 2) if not torch.cuda.is_available() else 0
if proc <= 8 and not torch.cuda.is_available():
    c.print(f"[dim red]Due to your device's hardware, generating responses may take a while. ({proc} workers)[/]")

tokenizer = transformers.AutoTokenizer.from_pretrained(model, skip_special_tokens=True)
conversation = [
    {
        'role': 'system',
        'content': 'You are Llama-3.2-3B, an AI assistant that answers briefly and factually.\
Do not add personal background, emotions or biographies unless asked.'
    }
]

while True:
    prompt = c.input("[bold]> [/]")
    conversation.append({'role': 'user', 'content': prompt})
    stream = transformers.TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    t: threading.Thread = use.generate(conversation,
                        stream,
                        max_new_tokens = 1024,
                        num_workers = proc,
                        temperature = 0.65,
                        top_p = 0.92,
                        top_k = 0,
                        repetition_penalty = 1.09,
                        no_repeat_ngram_size = 3,
                        eos_token_id=tokenizer.eos_token_id,  # sau valoarea integer corespunzătoare
                        pad_token_id=tokenizer.eos_token_id,  # sau valoarea integer corespunzătoare
                        )[0]
    response = ''
    for i in stream:
        if i == '<|eot_id|>':
            continue
        sys.stdout.write(i)
        sys.stdout.flush()
        response += i

    sys.stdout.write('\n')
    sys.stdout.flush()

    t.join()
    conversation.append({'role': 'assistant', 'content': response})
