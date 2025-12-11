import torch
import models.auto_register
from modelscope import AutoTokenizer, AutoModelForCausalLM
import os
import time
import numpy as np
import sys

# os.environ['FRIC_OFFLOAD'] = '1'

###############################################

model_id = "/mnt/data/xujiahao/FRIC/model"

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
CTX_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 32
print(f"Device: {device}, CTX_LEN: {CTX_LEN}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
).to(device)

raw = open("/mnt/data/xujiahao/FRIC/calibration_data_v5_rc.txt").read()
tokens = tokenizer.encode(raw)

messages = [
    {"role": "system", "content": "You are a kind chatbot."},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

generated_tokens = []
decode_times = []
prefill_times = []

is_first_time = True


def model_run():
    with torch.inference_mode():
        t00 = time.perf_counter()
        outputs = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        prefill_times.append(t1 - t00)

        logits = outputs.logits
        past_key_values = outputs.past_key_values
        print(type(past_key_values))
        generated_tokens = []
           
    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
    generated_tokens.append(next_token.item())
    model.model.fric_offloader.set_last_token_id(next_token)

    for step in range(56):
        with torch.inference_mode():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            decode_times.append(t1 - t0)
            logits = outputs.logits
            past_key_values = outputs.past_key_values
        
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
        generated_tokens.append(next_token.item())
        model.model.fric_offloader.set_last_token_id(next_token)
    
    response_ids = torch.cat([input_ids, torch.tensor(generated_tokens, device=input_ids.device).unsqueeze(0)], dim=1)
    print(tokenizer.decode(response_ids[0], skip_special_tokens=True))

def show(times, label="Time"):
    times = np.array(times)
    worst = np.max(times)
    best = np.min(times)
    percentile_50 = np.percentile(times, 50)
    percentile_90 = np.percentile(times, 90)
    mean = np.mean(times)
    print(f"{label}:")
    print(f"  Best      : {best * 1000:.3f} ms")
    print(f"  50% perc  : {percentile_50 * 1000:.3f} ms")
    print(f"  Mean      : {mean * 1000:.3f} ms")
    print(f"  90% perc  : {percentile_90 * 1000:.3f} ms")
    print(f"  Worst     : {worst * 1000:.3f} ms")
    print("")

def input_prepare(batch_size=1):
    a = 0
    nums = 0
    if batch_size > 1:
        while a + batch_size * CTX_LEN < len(tokens) and nums < 50:
            batch_src = [tokens[a + i*CTX_LEN : a + (i+1)*CTX_LEN] for i in range(batch_size)]
            a += batch_size * CTX_LEN

            messages_list = []
            for src in batch_src:
                messages = [
                    {"role": "system", "content": "You are a kind chatbot."},
                    {"role": "user", "content": src},
                ]
                messages_list.append(messages)
            input_ids = tokenizer.apply_chat_template(
                messages_list,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)
            yield input_ids
    
    else:
        a = 0
        nums = 0
        while a + CTX_LEN < len(tokens) and nums < 50:
            nums += 1
            src = tokens[a:a+CTX_LEN]
            messages = [
                {"role": "system", "content": "You are a kind chatbot."},
                {"role": "user", "content": src },
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            yield input_ids


def prefill_benchmark(batch_size=1):
    a = 0
    nums = 0
    for input_ids in input_prepare(batch_size):
        with torch.inference_mode():
            torch.cuda.synchronize()
            t00 = time.perf_counter()
            outputs = model(input_ids=input_ids, use_cache=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            prefill_times.append(t1 - t00)

            global is_first_time
            if is_first_time is True:
                print(f"the prefill tokens' number is {outputs.past_key_values[0][0].shape[2]}")
                is_first_time = False
        model.model.fric_offloader.offset = [0] * model.model.fric_offloader.num_layers
    show(prefill_times, "prefill time")
    # model.model.fric_offloader.mmap_to_pagecache()


def decode_benchmark(batch_size=1):
    gen = input_prepare(1)
    input_ids = next(gen)
    outputs = model(input_ids=input_ids, use_cache=True)
    logits = outputs.logits
    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
    # past_key_values = outputs.past_key_values
    next_token = next_token.repeat(batch_size, 1)
    print(next_token.shape)

    LENGTH_PER_TRIAL = 20
    for _ in range(LENGTH_PER_TRIAL):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(
            input_ids=next_token,
            use_cache=True,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        decode_times.append(t1 - t0)

    show(decode_times, "decode time")
        
def resume_test():
    next_token = input_ids
    past_key_values = None
    LENGTH_PER_TRIAL1 = 10
    for _ in range(LENGTH_PER_TRIAL1):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
        generated_tokens.append(next_token.item())
        model.model.fric_offloader.set_last_token_id(next_token)
    
    print("--RESUME TEST: STOP! tokens generated before the resume are below!--")
    response_ids = torch.cat([input_ids, torch.tensor(generated_tokens, device=input_ids.device).unsqueeze(0)], dim=1)
    print(tokenizer.decode(response_ids[0], skip_special_tokens=True))
    generated_tokens.clear()
    
    print("--RESUME TEST: CONTINUE! tokens generated after the resume are below!--")

    past_key_values = model.model.fric_offloader.sync_restore()

    LENGTH_PER_TRIAL2 = 47
    for _ in range(LENGTH_PER_TRIAL2):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
        generated_tokens.append(next_token.item())

    response_ids = torch.tensor(generated_tokens, device=input_ids.device).unsqueeze(0)
    print(tokenizer.decode(response_ids[0], skip_special_tokens=True))

def test_dtype_tansfer():
    for input_ids in input_prepare(1):
        outputs = model(input_ids=input_ids, use_cache=True)
        global is_first_time
        if is_first_time is True:
            print(f"the prefill tokens' number is {outputs.past_key_values[0][0].shape[2]}")
            is_first_time = False
            break
    print(model.model.fric_offloader.offset)
    pkv = model.model.fric_offloader.sync_restore_cpu(dtype=torch.bfloat16)
    print(pkv[0][0].shape)



    t0 = time.perf_counter()
    model.model.fric_offloader.sync_restore_cpu(dtype=torch.bfloat16)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    model.model.fric_offloader.sync_restore_cpu(dtype=torch.float32)
    t3 = time.perf_counter()

    print(f"FRIC sync restore CPU bfloat16 time: {(t1 - t0)*1000:.3f} ms")
    print(f"FRIC sync restore CPU float32 time: {(t3 - t2)*1000:.3f} ms")



if __name__ == "__main__":
    # model_run()

    prefill_benchmark(1)
    # decode_benchmark(512)

    # if os.environ.get('FRIC_OFFLOAD') == '1':
    #     model.model.fric_offloader.show()

    # print(model)
    # resume_test()

    # test_dtype_tansfer()