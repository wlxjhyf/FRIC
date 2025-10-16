import torch
import torch_tpu

import sys
import time
import os
from text_generation_server.utils.quantization import get_loader
from text_generation_server.models.custom_modeling.fric_soph_llama_modeling import (
    fric_FlashLlamaForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed_sophon,
    weight_files,
    Weights,
)
from text_generation_server.layers.attention import KVCache, Seqlen

from transformers import AutoTokenizer, AutoConfig

model_id = "/data/Llama-3.1-8B-Instruct"
device = sys.argv[1] if len(sys.argv) > 1 else "tpu"
CTX_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 32
dtype = torch.float16

####################  transformer Auto 尝试 ##################

# perfix = ""
# config = AutoConfig.from_pretrained(
#     model_id, revision=None, trust_remote_code=False
# )
# config.quantize = None
# config.speculator = None
# config.sliding_window = None
# weights_loader = get_loader(quantize = None, model_id = model_id, revision = None)
# filenames = weight_files(model_id, revision= None, extension=".safetensors")
# process_group, _, _, _ =  initialize_torch_distributed_sophon()
# weights = Weights(
#     filenames,
#     device,
#     dtype,
#     #process_group=torch.distributed.group.WORLD, # None
#     process_group=process_group,
#     aliases=None,
#     weights_loader=weights_loader,
# )

# tokenizer = AutoTokenizer.from_pretrained(
#     model_id,
#     revision=None,
#     padding_side="left",
#     truncation_side="left",
#     trust_remote_code=False,
# )
# tokenizer.pad_token = tokenizer.eos_token

# model = fric_FlashLlamaForCausalLM(perfix, config, weights)



# raw = open("/data/calibration_data_v5_rc.txt").read()
# tokens = tokenizer.encode(raw)


# messages = [
#     {"role": "system", "content": "You are a kind chatbot."},
#     {"role": "user", "content": "Who are you?"},
# ]

# input_ids = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt"
# ).to(device)

# generated_tokens = []
# decode_times = []
# prefill_times = []

# is_first_time = True


# def model_run():
#     t00 = time.perf_counter()
#     logits, speculative_logits = model(input_ids=input_ids)
#     torch.tpu.synchronize()
#     t1 = time.perf_counter()
#     prefill_times.append(t1 - t00)

#     past_key_values = model.kv_cache
#     generated_tokens = []
           
#     next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
#     generated_tokens.append(next_token.item())
#     # model.model.fric_offloader.set_last_token_id(next_token)

#     for step in range(56):
#         torch.tpu.synchronize()
#         t0 = time.perf_counter()
#         logits, _ = model(
#             input_ids = next_token,
#             kv_cacahe = past_key_values,
#         )
#         torch.tpu.synchronize()
#         t1 = time.perf_counter()
#         decode_times.append(t1 - t0)

#         past_key_values = model.kv_cache

#         next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
#         generated_tokens.append(next_token.item())
#         # model.model.fric_offloader.set_last_token_id(next_token)
    
#     response_ids = torch.cat([input_ids, torch.tensor(generated_tokens, device=input_ids.device).unsqueeze(0)], dim=1)
#     print(tokenizer.decode(response_ids[0], skip_special_tokens=True))




######### init ########
# forward需要很多参数，而这些参数大多在上一层被初始化
# 为了代码的简便性，坚决不用上层函数和batch
# 那么就需要对这些参数的生命周期进行管理
#'position_ids', 'cu_seqlen_prefill', 'kv_cache', 'block_tables', 'slots', 'seqlen', and 'max_s'

# 换策略，使用batch，但是不用generate
# 0层LM class，forward输入是batch，返回是logits
# 故我还需要去明白 logits 到 batch 的转换


# import math
# import soph_config
# from text_generation_server.models.globals import BLOCK_SIZE
# DECODE_TOKEN_LEN = soph_config.DECODE_TOKEN_LEN
# CONTEXT_LEN = soph_config.CONTEXT_LEN
# batches = 1
# chat_token_num = 27
# num_blocks = max(1024, (math.ceil((CONTEXT_LEN + DECODE_TOKEN_LEN + chat_token_num) / BLOCK_SIZE) * batches))
# cache_manager = model.init_kv_cache(
#     num_blocks,
#     model.num_layers,
#     model.num_kv_heads,
#     model.head_size,
#     model.dtype,
#     model.device,
# )
# position_ids = None
# cu_seqlen_prefill = None

# cache_lengths_tensor=None
# input_lengths_tensor=None

# # block_tables = 
# # slots = 

# # seqlen = Seqlen(
# #     input_lengths=input_lengths,
# #     cache_lengths=cache_lengths_tensor,
# #     cu_seqlen_q=cu_seqlen_prefill,
# #     max_q=batch.max_input_length,
# #     max_k=batch.max_current_length,
# # )

# # seqlen = 
# # max_s = 


#########################################################

# 先给出一版采用generate完成推理的代码 （妥协
# 但还是要摸清这部分关于CPU和GPU的控制，如果有很多的sync操作，要摸清其中的必要性
# 即排除serve之外的内容

# ----------- prepare ---------- #
from text_generation_server.models import Model, get_model
from text_generation_server.pb import generate_pb2
from text_generation_server.models.flash_causal_lm import FlashCausalLMBatch
import math
import soph_config
from text_generation_server.models.globals import BLOCK_SIZE
import numpy as np


def llama_chat_wrapper(question):
    return f"<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question}[/INST]"


def default_pb_parameters():
    return generate_pb2.NextTokenChooserParameters(
        temperature=1.0,
        repetition_penalty=1.0,
        top_k=0,
        top_p=1.0,
        typical_p=1.0,
        do_sample=False,
    )

def default_pb_stop_parameters(max_new_tokens=100):
    return generate_pb2.StoppingCriteriaParameters(
        stop_sequences=[], max_new_tokens=max_new_tokens
    )



DECODE_TOKEN_LEN = soph_config.DECODE_TOKEN_LEN
# CONTEXT_LEN = soph_config.CONTEXT_LEN
CONTEXT_LEN = 200
TEST_LEN = 50

batches = 1
chat_token_num = 27
mode="chat"
num_blocks = max(1024, (math.ceil((CONTEXT_LEN + DECODE_TOKEN_LEN + chat_token_num) / BLOCK_SIZE) * batches))


model = get_model(
    model_id=model_id,
    lora_adapter_ids=[],
    revision=None,
    sharded=False,
    quantize=None,
    speculate=None,
    dtype="float16",
    kv_cache_dtype=None,
    trust_remote_code=False,
    max_input_tokens=512,
)

cache_manager = model.init_kv_cache(
    num_blocks,
    model.num_layers,
    model.num_kv_heads,
    model.head_size,
    model.dtype,
    model.device,
)



def show(times, label="Time"):
    times = np.array(times)
    worst = np.max(times)
    best = np.min(times)
    percentile_50 = np.percentile(times, 50)
    mean = np.mean(times)
    print(f"{label}:")
    print(f"  Best      : {best / 1_000_000:.3f} ms")
    print(f"  50% perc  : {percentile_50 / 1_000_000:.3f} ms")
    print(f"  Mean      : {mean / 1_000_000:.3f} ms")
    print(f"  Worst     : {worst / 1_000_000:.3f} ms")
    print("")


def default_multi_batch_pb(batches, mode):
    questions = [
        "What is Deep Learning?I am new to this field. Please explain it with examples.",
    ] * batches
    question_length = [len(model.tokenizer.encode(q)) for q in questions]

    if mode == "chat":
        questions = [llama_chat_wrapper(question + ". " * max(0, CONTEXT_LEN - qlen - 27)) for qlen, question in zip(question_length, questions)]

    requests = []
    for i in range(batches):
        requests.append(
            generate_pb2.Request(
                id=i,
                inputs=questions[i],
                input_chunks=generate_pb2.Input(
                    chunks=[generate_pb2.InputChunk(text=questions[i])]
                ),
                prefill_logprobs=False,
                truncate=CONTEXT_LEN + 50,
                parameters=default_pb_parameters(),
                stopping_parameters=default_pb_stop_parameters(DECODE_TOKEN_LEN),
            )
        )
    batch_pb = generate_pb2.Batch(id=1, requests=requests, size=batches)
    return batch_pb


raw = open("/data/calibration_data_v5_rc.txt").read()
raw_tokens = raw

def input_prepare(batches = 1):
    a = 0
    nums = 0
    while a + CONTEXT_LEN < len(raw_tokens) and nums < TEST_LEN:
        nums += 1
        batch_src = raw_tokens[a : a + CTX_LEN]
        a += CTX_LEN
        questions = [
            batch_src,
        ] * batches
        question_length = [len(model.tokenizer.encode(q)) for q in questions]

        questions = [llama_chat_wrapper(question + ". " * max(0, CONTEXT_LEN - qlen - 27)) for qlen, question in zip(question_length, questions)]

        requests = []
        for i in range(batches):
            requests.append(
                generate_pb2.Request(
                    id=i,
                    inputs=questions[i],
                    input_chunks=generate_pb2.Input(
                        chunks=[generate_pb2.InputChunk(text=questions[i])]
                    ),
                    prefill_logprobs=False,
                    truncate=CONTEXT_LEN + 50,
                    parameters=default_pb_parameters(),
                    stopping_parameters=default_pb_stop_parameters(DECODE_TOKEN_LEN),
                )
            )
        batch_pb = generate_pb2.Batch(id=1, requests=requests, size=batches)
        next_batch = FlashCausalLMBatch.from_pb(
            batch_pb, model.tokenizer, model.dtype, model.device
        )
        yield next_batch



def model_run():
    time_list = []
    generated_text = {}

    batch_pb = default_multi_batch_pb(batches, mode)
    next_batch = FlashCausalLMBatch.from_pb(
        batch_pb, model.tokenizer, model.dtype, model.device
    )

    for i in range(DECODE_TOKEN_LEN):
        os.environ["TOKEN_IDX"] = str(i)

        generate_start = time.time_ns()
        generations, next_batch, (forward_ns, decode_ns) = model.generate_token(
            next_batch
        )
        generate_end = time.time_ns()
        
        time_list.append(generate_end - generate_start)

        for generation in generations:
            if i == 0:
                generated_text[generation.request_id] = generation.tokens.texts[0]
            else:
                generated_text[generation.request_id] += generation.tokens.texts[0]
        
        if next_batch is None:
                break
        

    for key in generated_text.keys():
        print(f"Batch {key}: {generated_text[key]}")
        print(f"FTL: {time_list[0] / 1000**2:.1f}ms, TPS: {batches / np.mean(time_list[1:]) * 1000**3:.1f}")
        print(f'TTFT: {time_list[0] /1000**3:.3f}s, TPOT: {np.mean(time_list[1:]) /1000**3:.3f}s, Throughput: {batches / np.mean(time_list[1:]) *1000**3:.1f}, TPS: {1/ np.mean(time_list[1:]) *1000**3:.1f}')

prefill_times = []
deocde_times = []


def prefill_benchmark():
    global prefill_times

    for next_batch in input_prepare():
        os.environ["TOKEN_IDX"] = str(0)

        generate_start = time.time_ns()
        _, _, _ = model.generate_token(
            next_batch
        )
        generate_end = time.time_ns()
        
        prefill_times.append(generate_end - generate_start)

    show(prefill_times)


def decode_benchmark():
    global deocde_times

    batch_pb = default_multi_batch_pb(batches, mode)
    next_batch = FlashCausalLMBatch.from_pb(
        batch_pb, model.tokenizer, model.dtype, model.device
    )
    _, next_batch, _ = model.generate_token(
        next_batch
    )

    for i in range(1, TEST_LEN):
        os.environ["TOKEN_IDX"] = str(i)

        generate_start = time.time_ns()
        _, _, _ = model.generate_token(
            next_batch
        )
        generate_end = time.time_ns()
        
        deocde_times.append(generate_end - generate_start)

    
    show(deocde_times)



if __name__ == "__main__":
    #model_run()
    prefill_benchmark()
    decode_benchmark()