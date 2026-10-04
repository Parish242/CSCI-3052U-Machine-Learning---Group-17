# Run notes: internlm2.5-7b full 2615 items run

## Model name and version
- internlm/internlm2_5-7b-chat

## Hardware
- Google Colab GPU: T4

## Quantization
- Quantization: bitsandbytes 8-bit (LLM.int8), float16 compute

## Library versions
- python 3.13.15, torch 2.11.0+cu130, CUDA (torch build) 13.0, transformers 4.51.3, tokenizers 0.21.4, accelerate 1.15.0, bitsandbytes 0.50.2, huggingface_hub 0.36.2, datasets 4.8.5, sentencepiece 0.2.1, scikit-learn 1.6.1, pandas 2.2.3, numpy 2.1.3

## Results
| condition | items | accuracy | macro F1 | weighted F1 | unparseable | minutes |
|---|---|---|---|---|---|---|
| without-rule | 2615 | 0.4340 | 0.3808 | 0.3855 | 0 | 17.4 |
| with-rule | 2615 | 0.6444 | 0.6126 | 0.6207 | 0 | 10.6 |

## Anything that went wrong
- Accuracy scores were within around 2-3 percentage points of the paper's results.