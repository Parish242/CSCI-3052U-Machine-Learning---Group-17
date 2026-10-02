# Run notes: internlm2.5-7b first 50 items trial run

## Model name and version
- internlm/internlm2_5-7b-chat

## Hardware
- Google Colab GPU: T4

## Quantization
- Quantization: bitsandbytes 8-bit (LLM.int8), float16 compute

## Library versions
- python 3.13.15, torch 2.11.0+cu130, CUDA (torch build) 13.0, transformers 4.51.3, tokenizers 0.21.4, accelerate 1.15.0, bitsandbytes 0.50.2, huggingface_hub 0.36.2, datasets 4.8.5, sentencepiece 0.2.1, scikit-learn 1.6.1, pandas 2.2.3, numpy 2.1.3

## Anything that went wrong
- The results of 70% (without RoT) and 96% (with RoT) on the first 50 items are different from the paper's 47% and 67.8% because the dataset is sorted by label, so all 50 trial items have the gold label `yes`
