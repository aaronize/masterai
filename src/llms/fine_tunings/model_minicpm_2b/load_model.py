"""

"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

path = 'openbmb/MiniCPM-2B-sft-bf16'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True)
print(f'[INFO] Load model[{path}] success!')


if __name__ == '__main__':
    from datetime import datetime
    while True:
        question = input('请输入问题：\n')
        start_time = datetime.now()
        response, history = model.chat(
            tokenizer,
            question,
            temperature=0.8,
            top_p=0.8
        )
        print('回答：\n', response)
        duration = (datetime.now() - start_time).seconds
        print(f'Duration: {duration}s')
