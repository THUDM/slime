from safetensors.torch import load_file

print(load_file("/root/slime_fsdp_test_checkpoint/1/model/model-00000-of-00000.safetensors").keys())
