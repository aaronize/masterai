"""

"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(device='mps', dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True)
model.eval()
print('>>> Load model success!')

if __name__ == '__main__':
    # image = Image.open('/Users/aaron/Workspace/masterai/src/llms/fine_tunings/model_minigpm_v_2/sh_OCR.jpeg').convert(
    #     'RGB')
    # question = 'Where is this photo taken?'
    question = 'Hello?'
    msg = [{'role': 'user', 'content': question}]

    answer, context, _ = model.chat(
        # image=image,
        msgs=msg,
        context=None,
        tokenizer=tokenizer,
        sampling=True
    )
    print('>>' * 20)
    print(answer)
