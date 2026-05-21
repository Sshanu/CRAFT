import os
import traceback
import llm
import utils

class PromptMutator:
    """
    Class for mutating prompts using predefined templates using few-shot examples.
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        with open("src/data/prompt_mutator/template.md", "r", encoding="utf-8") as f:
            self.template = f.read()

    def create_output_dir(self, output_dir, save_prefix):
        """
        Creates an output directory for saving results.
        """
        output_dir = os.path.join(output_dir, "mutator", save_prefix)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
            
    def process_candidate(self, system_prompt, conversation, user_message):
        """
        Adds the user message to the conversation.
        """
        overall_prompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        if conversation is not None:
            for turn in conversation:
                overall_prompt += f"<|im_start|>user\n{turn['User']}\n<|im_end|>\n<|im_start|>assistant\n{turn['Assistant']}\n<|im_end|>\n"
        overall_prompt += f"<|im_start|>user\n{user_message}\n<|im_end|>\n<|im_start|>assistant"
        return overall_prompt
    
    def llm_call(self, candidate, template, output_dir):
        """
        Calls the LLM with the given template and candidate.
        """
        token_usage = utils.TokenUsage()
        system_prompt, conversation, user_message = candidate.get_textual_prompt()
        template = template.replace("{prompt}", system_prompt)
    
        # Save the constructed template to a file
        with open(os.path.join(output_dir, "mutation_request.txt"), 'w', encoding="utf-8") as pf:
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
            token_usage.update_token_usage(token_usage_i)
            compressed_prompt = utils.extract_prompt_response(template_response_str)
            overall_prompt = self.process_candidate(compressed_prompt, conversation, user_message)
        except Exception as e:
            self.logger.log(f"Error while compressing candidate {candidate.id} : {e}")
            self.logger.log(traceback.format_exc(), False)
            overall_prompt = ""
            
        with open(os.path.join(output_dir, "mutation_reponse.txt"), 'w', encoding="utf-8") as rf:
            rf.write(overall_prompt)
            
        return overall_prompt
            
    def mutate_prompt(self, candidate, save_prefix, task_examples, output_dir):
        """
        Generates a prompt using the specified template and input-output examples.
        """
        token_usage = utils.TokenUsage()
        self.logger.log(f"Mutating candidate {candidate.id}")
        template = self.template.replace("{input_output_pairs}", task_examples)
        output_dir = self.create_output_dir(output_dir, f"candidate_{candidate.id}/{save_prefix}")
        mutated_prompt = self.llm_call(candidate, template, output_dir)
        return mutated_prompt, token_usage