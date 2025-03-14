# Description: This file contains the code to build the model and collator for each of the baseline models.

#================================================================
#                       Qwen-2.5-VL Model
#================================================================
class QwenCollator:
    def __init__(self, processor, args, process_vision_info, train=False):        
        multi_image_prompts = {
            "sequence_filling": "Pick the panel from the options that best fits the context space marked as MASK in the context. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "char_coherence": "Pick the panel from the options that best fits the context space marked as MASK in the context. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "visual_closure": "Pick the panel from the options that best fits the context space marked as MASK in the context. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "text_closure": "Pick the panel from the options that best fits the context space marked as MASK in the context. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "caption_relevance": "Pick the panel from the options that best follows the context caption. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
        }

        single_image_prompts = {
            "sequence_filling": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "char_coherence": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "visual_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "text_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "caption_relevance": "Pick A Panel Task: In the image you have a row of comic panels. From the options pick the panel that best follows the context caption. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
        }

        train_single_image_prompts = {
            "sequence_filling": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "char_coherence": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "visual_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "text_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "caption_relevance": "Pick A Panel Task: In the image you have a row of comic panels. From the options pick the panel that best follows the context caption. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
        }

        
        self.processor = processor
        self.train = train

        self.process_vision_info = process_vision_info
        if self.train:
            self.prompts = multi_image_prompts if not args.single_image else train_single_image_prompts
            self.build_message = self._build_train_message_multi_panel if not args.single_image else self._build_train_message_single_image
        else:
            #self.prompts = multi_image_prompts if not args.single_image else single_image_prompts
            self.prompts = multi_image_prompts if not args.single_image else train_single_image_prompts
            self.build_message = self._build_message_multi_panel if not args.single_image else self._build_message_single_image

    def _build_message_multi_panel(self, batch):
        messages = []
        for sample in batch:
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompts[sample["task_type"]]},
                    {"type": "text", "text": f"context: {sample['previous_panel_caption']}"}, # previous panel caption is empty in all skills except for caption_relevance 
                ],
            }
            i = 0
            for image in sample["context"]:
                message["content"].append({"type": "text", "text": f"\n{i}:"})
                if i == sample["index"]:
                    message["content"].append({"type": "text", "text": "MASK"})
                    i += 1
                    message["content"].append({"type": "text", "text": f"\n{i}:"})
                    message["content"].append({"type": "image", "image": image})
                else:
                    message["content"].append({"type": "image", "image": image})
                i += 1

            message["content"].append({"type": "text", "text": "\n\noptions: "})
            for i, image in enumerate(sample["options"]):
                message["content"].append({"type": "text", "text": f" \n{i}:"})
                message["content"].append({"type": "image", "image": image})

            messages.append([message])

        return messages

    def _build_message_single_image(self, batch):
            messages = []
            for sample in batch:
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompts[sample["task_type"]]},
                        {"type": "text", "text": f"context: {sample['previous_panel_caption']}"}, # previous panel caption is empty in all skills except for caption_relevance
                        {"type": "image", "image": sample["single_image"]},
                    ],
                }
                messages.append([message])

            return messages
    
    def _build_train_message_multi_panel(self, batch):
        raise NotImplementedError
    
    def _build_train_message_single_image(self, batch):
        messages = []
        for sample in batch:
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompts[sample["task_type"]]},
                    {"type": "text", "text": f"context: {sample['previous_panel_caption']}"}, # previous panel caption is empty in all skills except for caption_relevance
                    {"type": "image", "image": sample["single_image"]},
                ],
            }
            answer = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"answer: {sample['solution_index']}"},
                ],
            }
            messages.append([message, answer])

        return messages


    def __call__(self, batch):
        messages = self.build_message(batch)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False if self.train else True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side='left',
            truncation=True,
            return_tensors="pt",
            
        )

        if self.train:
            labels = inputs["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            image_tokens = [151652, 151653, 151654, 151655] # image tokens
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100

            inputs["labels"] = labels

            return inputs
        else:
            return inputs, dict(labels=[sample["solution_index"] for sample in batch], sample_ids=[sample["sample_id"] for sample in batch], messages=messages)

def build_qwen(args):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    if 'lora' in args.model:
        model_path = args.model.split('_lora')[0] if '_' in args.model else args.model.split('-lora')[0]
    else:
        model_path = args.model
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='auto' if args.max_steps == 0 else None
    )

    if 'lora' in args.model:
        model.load_adapter(args.model)

    processor = AutoProcessor.from_pretrained(model_path)
    collator = QwenCollator(processor, args, process_vision_info)

    if args.max_steps != 0:
        train_collator = QwenCollator(processor, args, process_vision_info, train=True)
        return model, collator, train_collator

    return model, collator


#================================================================
#                       SmolVLM Model
#================================================================
class SmolVLMCollator:
    def __init__(self, processor, args, train=False):
        
        multi_image_prompts = {
            "sequence_filling": "Pick the panel from the options that best fits the context space marked as MASK in the context. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "char_coherence": "Pick the panel from the options that best fits the context space marked as MASK in the context. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "visual_closure": "Pick the panel from the options that best fits the context space marked as MASK in the context. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "text_closure": "Pick the panel from the options that best fits the context space marked as MASK in the context. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "caption_relevance": "Pick the panel from the options that best follows the context caption. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
        }

        single_image_prompts = {
            "sequence_filling": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "char_coherence": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "visual_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "text_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "caption_relevance": "Pick A Panel Task: In the image you have a row of comic panels. From the options pick the panel that best follows the context caption. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
        }

        train_single_image_prompts = {
            "sequence_filling": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "char_coherence": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "visual_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "text_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "caption_relevance": "Pick A Panel Task: In the image you have a row of comic panels. From the options pick the panel that best follows the context caption. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
        }

        single_image_prompts_simple = {
            "sequence_filling": "Which panel from the bottom row best fits the missing panel spot in the top row? Reply with a number. \n\n",
            "char_coherence": "Which panel from the bottom row best fits the missing panel spot in the top row? Reply with a number. \n\n",
            "visual_closure": "Which panel from the bottom row best fits the missing panel spot in the top row? Reply with a number. \n\n",
            "text_closure": "Which panel from the bottom row best fits the missing panel spot in the top row? Reply with a number. \n\n",
            "caption_relevance": "Which panel best follows the context caption? Reply with a number. \n\n",
        }
        self.train = train

        self.create_images_list = self._create_images_list_multi_panel if not args.single_image else self._create_images_list_single_image
        self.processor = processor

        self.image_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')

        if self.train:
            self.prompts = multi_image_prompts if not args.single_image else train_single_image_prompts
            self.build_message = self._build_train_message_multi_panel if not args.single_image else self._build_train_message_single_image
        else:
            #self.prompts = multi_image_prompts if not args.single_image else single_image_prompts
            self.prompts = multi_image_prompts if not args.single_image else train_single_image_prompts
            self.build_message = self._build_message_multi_panel if not args.single_image else self._build_message_single_image

    def _create_images_list_multi_panel(self, batch):
        images = []
        for sample in batch:
            images.extend(sample["context"])
            images.extend(sample["options"])

            del sample["context"]
            del sample["options"]
        
        return images


    def _create_images_list_single_image(self, batch):
        return [sample["single_image"] for sample in batch]

    def _build_message_multi_panel(self, batch):
        messages = []
        for sample in batch:
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompts[sample["task_type"]]},
                    {"type": "text", "text": f"context: {sample['previous_panel_caption']}"}, # previous panel caption is empty in all skills except for caption_relevance 
                ],
            }
            i = 0
            for image in sample["context"]:
                message["content"].append({"type": "text", "text": f"\n{i}:"})
                if i == sample["index"]:
                    message["content"].append({"type": "text", "text": "MASK"})
                    i += 1
                    message["content"].append({"type": "text", "text": f"\n{i}:"})
                    message["content"].append({"type": "image"})
                else:
                    message["content"].append({"type": "image"})
                i += 1

            message["content"].append({"type": "text", "text": "\n\noptions: "})
            for i, image in enumerate(sample["options"]):
                message["content"].append({"type": "text", "text": f" \n{i}:"})
                message["content"].append({"type": "image"})

            messages.append([message])

        return messages

    def _build_message_single_image(self, batch):
            messages = []
            for sample in batch:
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompts[sample["task_type"]]},
                        {"type": "text", "text": f"context: {sample['previous_panel_caption']}"}, # previous panel caption is empty in all skills except for caption_relevance
                        {"type": "image"},
                    ],
                }
                messages.append([message])

            return messages
    
    def _build_train_message_single_image(self, batch):
        messages = []
        for sample in batch:
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompts[sample["task_type"]]},
                    {"type": "text", "text": f"context: {sample['previous_panel_caption']}"}, # previous panel caption is empty in all skills except for caption_relevance
                    {"type": "image"},
                ],
            }
            answer = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"answer: {sample['solution_index']}"},
                ],
            }
            messages.append([message, answer])

        return messages

    def __call__(self, batch):
        messages = self.build_message(batch)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=text,
            images=self.create_images_list(batch),
            padding=True,
            padding_side='left',
            truncation=True,
            return_tensors="pt",
            
        )

        if self.train:
            labels = inputs["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            image_tokens = [self.image_token_id] # image tokens
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100

            inputs["labels"] = labels

            return inputs
        else:
            return inputs, dict(labels=[sample["solution_index"] for sample in batch], sample_ids=[sample["sample_id"] for sample in batch], messages=messages)

def build_smolvlm(args):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        device_map='auto' if args.max_steps == 0 else None
    )
    collator = SmolVLMCollator(processor, args)

    if args.max_steps != 0:
        train_collator = SmolVLMCollator(processor, args, train=True)
        return model, collator, train_collator
    
    return model, collator


#================================================================
#                       Llama Vision Model
#================================================================
class LlamaCollator:
    def __init__(self, processor, args):
        
        multi_image_prompts = {}

        single_image_prompts = {
            "sequence_filling": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "char_coherence": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "visual_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "text_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "caption_relevance": "Pick A Panel Task: In the image you have a row of comic panels. From the options pick the panel that best follows the context caption. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
        }

        self.prompts = multi_image_prompts if not args.single_image else single_image_prompts
        self.processor = processor

        self.build_message = self._build_message_multi_panel if not args.single_image else self._build_message_single_image
        self.create_images_list = self._create_images_list_multi_panel if not args.single_image else self._create_images_list_single_image

    def _create_images_list_multi_panel(self, batch):
        images = []
        for sample in batch:
            images.extend(sample["context"])
            images.extend(sample["options"])
        
        return images


    def _create_images_list_single_image(self, batch):
        return [sample["single_image"] for sample in batch]

    def _build_message_multi_panel(self, batch):
        raise NotImplementedError("Llama model does not support multi images")

    def _build_message_single_image(self, batch):
            messages = []
            for sample in batch:
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompts[sample["task_type"]]},
                        {"type": "text", "text": f"context: {sample['previous_panel_caption']}"}, # previous panel caption is empty in all skills except for caption_relevance
                        {"type": "image"},
                    ],
                }
                messages.append([message])

            return messages

    def __call__(self, batch):
        messages = self.build_message(batch)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=text,
            images=self.create_images_list(batch),
            padding=True,
            padding_side='left',
            truncation=True,
            return_tensors="pt",
            
        )
        return inputs, dict(labels=[sample["solution_index"] for sample in batch], sample_ids=[sample["sample_id"] for sample in batch], messages=messages)

def build_llama(args):
    import torch
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
    model = MllamaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    ).to("cuda")
    collator = LlamaCollator(processor, args)

    return model, collator


#================================================================
#                       Paligemma Model
#================================================================

class PaliGemmaCollator:
    def __init__(self, processor, args, train=False):        


        single_image_prompts = {
            "sequence_filling": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "char_coherence": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "visual_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "text_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "caption_relevance": "Pick A Panel Task: In the image you have a row of comic panels. From the options pick the panel that best follows the context caption. You can reason about your answer, but you must return your final answer as a number with 'answer: <your answer here>'\n\n",
        }

        self.processor = processor
        self.train = train

        self.prompts = single_image_prompts


    def __call__(self, batch):
        messages = []
        for sample in batch:
            message = self.prompts[sample["task_type"]]
            if sample["task_type"] == "caption_relevance":
                message.format(caption=sample["previous_panel_caption"])
            messages.append(message)
        inputs = self.processor(text=messages, images=[sample["single_image"] for sample in batch], return_tensors="pt", padding="longest")

        return inputs, dict(labels=[sample["solution_index"] for sample in batch], sample_ids=[sample["sample_id"] for sample in batch], messages=messages)

def build_paligemma(args):
    import torch
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='auto' if args.max_steps == 0 else None
    )

    processor = PaliGemmaProcessor.from_pretrained(args.model)
    collator = PaliGemmaCollator(processor, args)

    return model, collator


#================================================================
#                       Molmo Model
#================================================================

class MolmoCollator:
    def __init__(self, processor, args, train=False):        


        single_image_prompts = {
            "sequence_filling": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "char_coherence": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "visual_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "text_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "caption_relevance": "Pick A Panel Task: In the image you have a row of comic panels. From the options pick the panel that best follows the context caption. You must return your final answer as a number with 'answer: <your answer here>'",
        }

        self.processor = processor
        self.train = train

        self.prompts = single_image_prompts


    def __call__(self, batch):
        assert len(batch) == 1, "Molmo only supports single sample batches. Why? Beats me, I started modifying their processor to support batches, but I was too lazy to finish it."
        
        sample = batch[0]

        message = self.prompts[sample["task_type"]]
        if sample["task_type"] == "caption_relevance":
            message += f"context: {sample["previous_panel_caption"]}\n\n"
        
        inputs = self.processor.process(text=message, images=[sample["single_image"]], return_tensors="pt", padding="longest")

        return inputs, dict(labels=[sample["solution_index"] for sample in batch], sample_ids=[sample["sample_id"] for sample in batch])

def build_molmo(args):
    import transformers
    assert transformers.__version__ == "4.43.3", "Molmo does not work with newer versions of transformers"

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map='auto' if args.max_steps == 0 else None,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    collator = MolmoCollator(processor, args)

    return model, collator

#================================================================
#                       MiniCPM Model
#================================================================

class MiniCPMCollator:
    def __init__(self, processor, args, train=False):        


        single_image_prompts = {
            "sequence_filling": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "char_coherence": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "visual_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "text_closure": "Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n",
            "caption_relevance": "Pick A Panel Task: In the image you have a row of comic panels. From the options pick the panel that best follows the context caption. You must return your final answer as a number with 'answer: <your answer here>'",
        }

        self.processor = processor
        self.train = train

        self.prompts = single_image_prompts


    def __call__(self, batch):

        assert len(batch) == 1, "MiniCPM model only supports single sample batches. Why? I don't know!!!!! People use batches right? Or am I the crazy one?"
        
        sample = batch[0]
        
        message = self.prompts[sample["task_type"]]
        if sample["task_type"] == "caption_relevance":
            message += f"context: {sample["previous_panel_caption"]}\n\n"
        message = [{'role': 'user', 'content': message}]

        images = sample["single_image"]

        inputs = {
            "msgs": message,
            "image": images,
            "context": None
        }

        return inputs, dict(labels=[sample["solution_index"] for sample in batch], sample_ids=[sample["sample_id"] for sample in batch], messages=message)

def build_minicpm(args):
    import transformers
    assert transformers.__version__ == "4.36.0", "MiniCPM does not work with newer versions of transformers"

    import torch
    from transformers import AutoModel, AutoProcessor
    
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="flash_attention_2",
        device_map='auto' if args.max_steps == 0 else None,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    collator = MiniCPMCollator(processor, args)

    return model, collator

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hf_repo", type=str, required=True)
    args = parser.parse_args()

    args.max_steps = 0
    args.single_image = True

    if "qwen" in args.model.lower():
        model, collator = build_qwen(args)
    elif "smolvlm" in args.model.lower():
        model, collator = build_smolvlm(args)
    elif "llama" in args.model.lower():
        model, collator = build_llama(args)
    elif "paligemma" in args.model.lower():
        model, collator = build_paligemma(args)
    elif "molmo" in args.model.lower():
        model, collator = build_molmo(args)
    elif "minicpm" in args.model.lower():
        model, collator = build_minicpm(args)
    else:
        raise ValueError("Model not supported")

    model.push_to_hub(args.hf_repo)