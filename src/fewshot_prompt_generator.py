import traceback
import llm
import utils

class FewShotPromptGenerator:
    """
    Class for generating prompts using predefined templates using few-shot examples.
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        with open("src/data/prompt_generator/long_template.md", "r", encoding="utf-8") as f:
            self.long_template = f.read()
        with open("src/data/prompt_generator/medium_template.md", "r", encoding="utf-8") as f:
            self.medium_template = f.read()
        with open("src/data/prompt_generator/short_template.md", "r", encoding="utf-8") as f:
            self.short_template = f.read()
            
    def generate_prompt(self, usermessage, prompt_type, task_examples, output_dir):
        """
        Generates a prompt using the specified template and input-output examples.

        Args:
            prompt_type (str): The type of prompt to generate (e.g., 'critic', 'actor').
            input_output_examples (list): A list of dictionaries with 'input' and 'output' keys.

        Returns:
            str: The generated prompt.
        """
        token_usage = utils.TokenUsage()
        self.logger.log(f"Generating {prompt_type} prompt")
        
        if prompt_type == "long":
            template = self.long_template
        elif prompt_type == "medium":
            template = self.medium_template
        elif prompt_type == "short":
            template = self.short_template
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        template = template.replace("{input_output_pairs}", task_examples)
        
        # Save the constructed template to a file
        with open(output_dir + f"prompt_generation_{prompt_type}_request.txt", 'w', encoding="utf-8") as pf:
            pf.write(template)
            
        try:
            _, template_response_str, token_usage_i = llm.infer(
                template,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                model=self.config.model,
                provider=self.config.provider
            )
            generated_prompt = utils.extract_prompt_response(template_response_str)
            generated_prompt = "<|im_start|>system\n" + generated_prompt + "\n<|im_end|>\n<|im_start|>user\n" + usermessage + "\n<|im_end|>\n<|im_start|>assistant"
        except Exception as e:
            self.logger.log(f"Error while generating prompt with {prompt_type}: {e}")
            self.logger.log(traceback.format_exc(), False)
            generated_prompt = ""
        
        with open(output_dir + f"prompt_generation_{prompt_type}_response.txt", 'w', encoding="utf-8") as rf:
            rf.write(generated_prompt)
        return generated_prompt, token_usage
