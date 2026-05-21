import abc
import os
import traceback
import llm
import utils
import prompt_template
from typing import Optional

class CompressionStrategy(abc.ABC):
    """Abstract base class for prompt compression strategies."""
    def __init__(self, strategy_name, config, logger, config_orch, temp_dir):
        """
        Initializes the compression strategy with configuration and logger.
        """
        self.strategy_name = strategy_name
        self.config = config
        self.logger = logger
        self.config_orch = config_orch
        self.temp_dir = temp_dir
    
    def create_output_dir(self, output_dir, save_prefix):
        """
        Creates an output directory for saving results.
        """
        output_dir = os.path.join(output_dir, "compressor", self.strategy_name, save_prefix)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @abc.abstractmethod
    def compress(self, candidate, train_batch, compression_ratio: float, output_dir: str, new_prompt_id:int, current_round:int) -> Optional['prompt_template.PromptTree']:
        """
        Applies the compression strategy to the given prompt.

        Args:
            prompt: The input prompt string.

        Returns:
            The compressed prompt string.
        """
        pass
    
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
    
    def llm_call(self, candidate, template, output_dir, compression_ratio):
        """
        Calls the LLM with the given template and candidate.
        """
        output_dir = os.path.join(self.create_output_dir(output_dir, f"candidate_{candidate.id}"), f"ratio_{compression_ratio}")
        token_usage = utils.TokenUsage()
        system_prompt, conversation, user_message = candidate.get_textual_prompt()
        template = template.replace("{prompt}", system_prompt)
    
        # Save the constructed template to a file
        with open(os.path.join(output_dir, f"candidate_{candidate.id}_compression_request.txt"), 'w', encoding="utf-8") as pf:
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
            overall_prompt = compressed_prompt
            overall_prompt = self.process_candidate(compressed_prompt, conversation, user_message)
        except Exception as e:
            self.logger.log(f"Error while compressing candidate {candidate.id} : {e}")
            self.logger.log(traceback.format_exc(), False)
            overall_prompt = ""
            
        with open(os.path.join(output_dir, f"candidate_{candidate.id}_compression_reponse.txt"), 'w', encoding="utf-8") as rf:
            rf.write(overall_prompt)
            
        return overall_prompt

    def _create_prompt_tree(self, text, tree_id, parent_id, round_created):
        """
        Creates a single PromptTree object.
        """
        tree = prompt_template.PromptTree(
            tree_id,
            text,
            self.config_orch.model,
            self.config_orch.provider,
            self.temp_dir,
            parent_id=parent_id,
            round_created=round_created,
            formatting=self.config_orch.format_prompt,
        )
        return tree