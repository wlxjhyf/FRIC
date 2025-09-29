from transformers import AutoModelForCausalLM, LlamaConfig, AutoConfig
from .llama import fric_LlamaForCausalLM
from .configuration_llama import fric_LlamaConfig


AutoConfig.register("fric_llama", fric_LlamaConfig)
AutoModelForCausalLM.register(fric_LlamaConfig, fric_LlamaForCausalLM)