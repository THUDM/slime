from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/root/slime_fsdp_test_checkpoint/1", subfolder="model")
