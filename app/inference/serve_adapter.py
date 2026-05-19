import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from app.training.prompt_templates import build_inference_prompt
from app.core.secrets import apply_secrets_to_environment, get_hf_token


def generate_response(
    base_model: str,
    adapter_repo: str,
    prompt: str,
    input_text: str = "",
    prompt_template: str = "alpaca",
    max_new_tokens: int = 128,
) -> str:
    apply_secrets_to_environment()

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=get_hf_token())

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        token=get_hf_token(),
    )

    model = PeftModel.from_pretrained(model, adapter_repo, token=get_hf_token())
    model.eval()

    formatted_prompt = build_inference_prompt(prompt, input_text, prompt_template)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()
