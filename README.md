# Comics Pick-A-Panel
Code for the Comics Pick-a-Panel dataset and baselines 

## Dataset
The dataset is available at HuggingFace: [VLR-CVC/ComicsPAP](https://huggingface.co/datasets/VLR-CVC/ComicsPAP)

## Eval
```python
python evaluate.py --split val --skill sequence_filling char_coherence visual_closure text_closure caption_relevance --model PATH_TO_MODEL --eval_batch_size BATCH_SIZE --dataset_cache PATH_TO_SAVE_DATASET --single_image
``` 

## Train
```python
python sft.py --skill sequence_filling char_coherence visual_closure text_closure caption_relevance --model PATH_TO_MODEL --batch_size BATCH_SIZE --max_steps TOTAL_STEPS --eval_steps EVAL_AND_SAVE_STEPS --dataset_cache PATH_TO_SAVE_DATASET --single_image
```


