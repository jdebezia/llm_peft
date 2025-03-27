from abc import abstractmethod, ABC
import json

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

## Abstract class acting as port for language model
class LargeLanguageModel(ABC):

    @abstractmethod
    def generate(query: str) -> str:
        pass

## Qwen2 language model class based on languageModel port
class LLM:
    def __init__(self, model: AutoPeftModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, natural_query: str) -> str:

        text = self.tokenizer.apply_chat_template(
            natural_query,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.9
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split("SQLQuery:")[-1]

## prepare query with prompt
def prepare_query(natural_query: str, path_to_prompt_template: str) -> str:
    template = json.load(open(path_to_prompt_template, "r"))
    prompt = f"{template['role']} \n --{template['context']}"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": natural_query}
    ]
    return messages


## generate sql from natural query with given llm
def generate_sql(natural_query: str, language_model: LargeLanguageModel) -> str:
    return language_model.generate(natural_query)