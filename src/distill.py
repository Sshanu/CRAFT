import json
import traceback
from tqdm import tqdm
import dirtyjson
from collections import defaultdict
import collections, itertools
import heapq
from sentence_transformers import SentenceTransformer, util
import numpy as np
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
import os

# Local modules
import utils
import llm
from base_compressor import CompressionStrategy

class Distill(CompressionStrategy):
    """
    Handles prompt compression using a critic-actor model.
    """
    def __init__(self, strategy_name, config, logger, config_orch, temp_dir):
        super().__init__(strategy_name, config, logger, config_orch, temp_dir)
        with open("src/data/prompt_compressor/distill/critic_template.md", "r", encoding="utf-8") as f:
            self.critic_template = f.read()
        with open("src/data/prompt_compressor/distill/actor_template.md", "r", encoding="utf-8") as f:
            self.actor_template = f.read()

    def compress(self, candidate, train_batch, compression_ratio: float, output_dir: str, new_prompt_id:int, current_round:int):
        """
        Compresses a prompt to a target compression ratio.
        """
        output_dir = os.path.join(self.create_output_dir(output_dir, f"candidate_{candidate.id}"), f"ratio_{compression_ratio}")
        self.logger.log(f"Output directory for compression: {output_dir}")

        compression_target = 100 * compression_ratio
        self.logger.log(f"Starting prompt compression for candidate {candidate.id} with target {compression_target}")
        
        try:
            # Run critic to get compression feedback
            critic_feedback, _ = self.run_critic(candidate, compression_target, train_batch, output_dir)

            if not critic_feedback:
                self.logger.log("No feedback from critic. Skipping compression.")
                return None

            # Run actor to apply compression
            compressed_candidate, _ = self.run_actor(candidate, critic_feedback, train_batch, output_dir, new_prompt_id, current_round)

            return compressed_candidate

        except Exception as e:
            self.logger.log(f"Error during prompt compression: {str(e)}")
            self.logger.log(traceback.format_exc(), False)
            return None

    def run_critic(self, candidate, compression_target, input_output_pairs, output_dir):
        """
        Runs the compression critic to get feedback on how to compress the prompt.
        """
        token_usage_critic = utils.TokenUsage()
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.log("Running compression critic")
            # Assuming tree_to_json_with_tokens() exists in the PromptTree class
            prompt_tree_json = json.dumps(candidate.tree_to_json_with_tokens(), indent=4)
            input_output_pairs_str = json.dumps(input_output_pairs, indent=4)

            critic_prompt = self.critic_template.replace("{parsed_prompt}", prompt_tree_json)
            critic_prompt = critic_prompt.replace("{compression_target}", str(compression_target))
            critic_prompt = critic_prompt.replace("{input_output_pairs}", input_output_pairs_str)

            with open(os.path.join(output_dir, "compression_critic_prompt.txt"), 'w', encoding="utf-8") as pf:
                pf.write(critic_prompt)

            _, critic_response_str, token_usage_i = llm.infer(
                critic_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                model=self.config.model,
                provider=self.config.provider
            )
            token_usage_critic.update_token_usage(token_usage_i)
            critic_response_str = utils.clean_llm_response(critic_response_str)

            with open(os.path.join(output_dir, "compression_critic_response.txt"), 'w', encoding="utf-8") as rf:
                rf.write(critic_response_str)

            critic_feedback = dirtyjson.loads(critic_response_str)
            return critic_feedback, token_usage_critic

        except Exception as e:
            self.logger.log(f"Error in compression critic: {e}")
            self.logger.log(traceback.format_exc(), False)
            return None, token_usage_critic

    def run_actor(self, candidate, critic_feedback, input_output_pairs, output_dir, new_prompt_id, current_round):
        """
        Runs the compression actor to apply feedback and compress the prompt.
        """
        token_usage_actor = utils.TokenUsage()
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.log("Running compression actor")
            prompt_tree_json = json.dumps(candidate.tree_to_json_with_tokens(), indent=4)
            feedback_str = json.dumps(critic_feedback, indent=4)
            input_output_pairs_str = json.dumps(input_output_pairs, indent=4)

            actor_prompt = self.actor_template.replace("{parsed_prompt}", prompt_tree_json)
            actor_prompt = actor_prompt.replace("{critic_feedback}", feedback_str)
            actor_prompt = actor_prompt.replace("{input_output_pairs}", input_output_pairs_str)

            with open(os.path.join(output_dir, "compression_actor_prompt.txt"), 'w', encoding="utf-8") as f:
                f.write(actor_prompt)

            _, actor_response_str, token_usage_i = llm.infer(
                actor_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                model=self.config.model,
                provider=self.config.provider
            )
            token_usage_actor.update_token_usage(token_usage_i)
            actor_response_str = utils.clean_llm_response(actor_response_str)

            with open(os.path.join(output_dir, "compression_actor_response.txt"), 'w', encoding="utf-8") as f:
                f.write(actor_response_str)

            actions = dirtyjson.loads(actor_response_str)
            new_candidate = self.apply_actions(candidate, actions["Actions"])
            self.logger.log(f"Actor updating candidate {candidate.id} with new prompt ID {new_prompt_id}")
            new_candidate.id = new_prompt_id
            new_candidate.parent_id = candidate.id
            new_candidate.round_created = current_round
            
            # Save the compressed prompt
            with open(os.path.join(output_dir, f"compressed_structured_candidate_{new_candidate.id}.txt"), 'w', encoding="utf-8") as f:
                f.write(json.dumps(new_candidate.get_json_prompt(), indent=4))
            with open(os.path.join(output_dir, f"compressed_candidate_{new_candidate.id}.txt"), 'w', encoding="utf-8") as f:
                f.write(str(new_candidate.get_prompt_chat_template()))

            return new_candidate, token_usage_actor

        except Exception as e:
            self.logger.log(f"Error in compression actor: {e}")
            self.logger.log(traceback.format_exc(), False)
            raise Exception("Error in running compression actor") from e

    def apply_actions(self, candidate, actions):
        """
        Applies a list of compression actions to a copied PromptTree instance.
        """
        updated_candidate = candidate.copy_and_reset()
        for action in actions:
            try:
                action_type = action["Action"]
                params = action["Params"]
                self.logger.log(f"Applying compression action: {action_type} on {params.get('NodeID', 'N/A')}")

                if action_type == "UPDATE_NODE_VALUE":
                    updated_candidate.update_node_value(params["NodeID"], params["NewValue"])
                elif action_type == "DELETE_NODE":
                    updated_candidate.delete_node(params["NodeID"])
                elif action_type == "MERGE_NODES":
                    updated_candidate.merge_nodes(
                        params["ParentID"],
                        params["NodeID1"],
                        params["NodeID2"],
                        params["MergedContent"]
                    )
                elif action_type == "UPDATE_SUBTREE":
                    # NOTE: This assumes a method `update_subtree` exists on the PromptTree object
                    # that can take a node ID and a string representing the new subtree,
                    # parse the string, and replace the old subtree.
                    # This is a complex operation and its implementation is not provided here.
                    updated_candidate.update_subtree(params["NodeID"], params["NewSubtree"])

            except Exception as e:
                self.logger.log(f"Error applying action {action.get('Action', 'N/A')}: {e}")
                self.logger.log(traceback.format_exc(), False)
        return updated_candidate