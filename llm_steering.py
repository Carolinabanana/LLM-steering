import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from sae import Sae

sae_path = "PATH"
device = "cuda:0"
sae = Sae.load_from_disk(sae_path).to(device)

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
prompt = "The capital of France is "
model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
model = LlamaForCausalLM.from_pretrained("NousResearch/Meta-Llama-3-8B", device_map=device, torch_dtype=torch.bfloat16)
torch.manual_seed(42)
out_features = []

#Modify layer 16, feature N, with scale 1.1
sae_args = {"sae": sae, "layer_to_modify":16, "feature_index":-1,"scale_factor":1.1,"out_features":out_features}

model_inputs["sae_args"] = sae_args

sample_output = model.generate(
        **model_inputs,
        max_new_tokens=20,
        do_sample=True,
        top_k=40,
        temperature=0.6,
        pad_token_id=tokenizer.eos_token_id,
    )

#Get top five features from modelling_llama output across the input tokens
print(out_features)
print("Baseline", tokenizer.decode(sample_output[0], skip_special_tokens=True))

#For those features, view the output text with them amplified
for feature_index in range(0,5):
    feature = out_features[0][feature_index]
    sae_args["feature_index"] = feature
    sample_output = model.generate(
        **model_inputs,
        max_new_tokens=20,
        do_sample=False,
    )

    print("Steering", feature, tokenizer.decode(sample_output[0], skip_special_tokens=True))

    
