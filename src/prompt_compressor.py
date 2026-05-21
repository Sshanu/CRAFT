import abc
import os
from typing import List, Optional
import traceback
import llm
import utils
from llmlingua import PromptCompressor as LLMLinguaCompressor
import dirtyjson
import json
from base_compressor import CompressionStrategy
from distill import Distill
import prompt_template

class PercentageCompressionStrategy(CompressionStrategy):
    """
    A compression strategy that uses an LLM to compress the prompt to a target percentage.
    """
    def __init__(self, strategy_name, config, logger, config_orch, temp_dir):
        super().__init__(strategy_name, config, logger, config_orch, temp_dir)
        with open("src/data/prompt_compressor/percentage_compression_template.md", "r", encoding="utf-8") as f:
            self.template = f.read()

    def compress(self, candidate, train_batch, compression_ratio: float, output_dir: str, new_prompt_id:int, current_round:int) -> Optional['prompt_template.PromptTree']:
        """
        Compresses the prompt using the compression_ratio.
        """
        compression_percentage = compression_ratio * 100
        template = self.template.replace("{compression_percentage}", str(compression_percentage))
        overall_prompt = self.llm_call(candidate, template, output_dir, compression_ratio)
        if not overall_prompt:
            return None
        return self._create_prompt_tree(
            text=overall_prompt,
            tree_id=new_prompt_id,
            parent_id=candidate.id,
            round_created=current_round,
        )

class LLMLingua(CompressionStrategy):
    def __init__(self, strategy_name, config, logger, config_orch, temp_dir):
        super().__init__(strategy_name, config, logger, config_orch, temp_dir)
        self.compressor = LLMLinguaCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cpu"
        )
        
    def compress(self, candidate, train_batch, compression_ratio: float, output_dir: str, new_prompt_id:int, current_round:int) -> Optional['prompt_template.PromptTree']:
        """
        Compresses the prompt using the compression_ratio.
        """
        system_prompt, conversation, user_message = candidate.get_textual_prompt()
        compressed_prompt = self.compressor.compress_prompt_llmlingua2(system_prompt, rate=1 - compression_ratio)['compressed_prompt']
        overall_prompt = self.process_candidate(compressed_prompt, conversation, user_message)
        output_dir = os.path.join(self.create_output_dir(output_dir, f"candidate_{candidate.id}"), f"ratio_{compression_ratio}")
        with open(f"{output_dir}_compression_reponse.txt", 'w', encoding="utf-8") as rf:
            rf.write(overall_prompt)

        if not overall_prompt:
            return None
        return self._create_prompt_tree(
            text=overall_prompt,
            tree_id=new_prompt_id,
            parent_id=candidate.id,
            round_created=current_round,
        )

class PromptCompressor:
    """
    A class for compressing prompts using various strategies.
    """

    def __init__(self, config, logger, config_orch, temp_dir):
        """
        Initializes the PromptCompressor with a list of compression strategies.

        Args:
            strategies: A list of CompressionStrategy instances to apply.
        """
        self.config = config
        self.logger = logger
        self.config_orch = config_orch
        self.temp_dir = temp_dir
        self.methods = self.config.methods.split(",")
        self.logger.log(f"Using compression strategies: {self.methods}")
        
        # Initialize the compression strategies
        self.strategies = []
        for method in self.methods:
            if method == "percentage":
                self.strategies.append((method, PercentageCompressionStrategy("percentage", self.config, self.logger, self.config_orch, self.temp_dir)))
            elif method == "llmlingua":
                self.strategies.append((method, LLMLingua("llmlingua", self.config, self.logger, self.config_orch, self.temp_dir)))
            elif method == "distill":
                self.strategies.append((method, Distill("distill", self.config, self.logger, self.config_orch, self.temp_dir)))
            else:
                self.logger.log(f"Unknown compression strategy: {method}")
                
    def compress(self, candidate, train_batch, compression_ratio, current_round: int, round_dir: str, new_prompt_id_start: int):
        """
        Compresses a candidate using all configured strategies.
        """
        compressed_trees = []
        for i, (method, strategy) in enumerate(self.strategies):
            new_prompt_id = new_prompt_id_start + i
            
            tree = strategy.compress(
                candidate=candidate,
                train_batch=train_batch,
                compression_ratio=compression_ratio,
                output_dir=round_dir,
                new_prompt_id=new_prompt_id,
                current_round=current_round
            )
            if tree:
                compressed_trees.append(tree)
        return compressed_trees