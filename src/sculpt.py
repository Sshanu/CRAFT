import os
import json
import traceback
from tqdm import tqdm
import dirtyjson
import random
from collections import defaultdict
import collections, itertools
import heapq
from sentence_transformers import SentenceTransformer, util
import numpy as np
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# Local modules
import utils
import llm

class Sculpt:
    """
    Main class for the SCULPT algorithm, responsible for generating and evolving prompts.
    """
    def __init__(self, config, logger, random_seed=42):
        self.strategy_name = "sculpt"
        self.config = config
        self.logger = logger
        self.encoder = SentenceTransformer(self.config.encoder_model, device=device)    
        
        # Create seeded random generator for reproducible SCULPT operations
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed    
        
        # Load templates for structural critic, error critic, and actor
        with open("src/data/prompt_enhancer/sculpt/structural_critic_template.md", "r", encoding="utf-8") as f:
            self.structural_critic_template = f.read()
        with open("src/data/prompt_enhancer/sculpt/error_critic_template.md", "r", encoding="utf-8") as f:
            self.error_critic_template = f.read()
        with open("src/data/prompt_enhancer/sculpt/actor_template.md", "r", encoding="utf-8") as f:
            self.actor_template = f.read()
            
        # Initialize variables
        self.current_round = 0
        self.max_rounds = 0
        self.prompt_universe_size = None
        
    def create_output_dir(self, output_dir, save_prefix):
        """
        Creates an output directory for saving results.
        """
        output_dir = os.path.join(output_dir, "enchancer", self.strategy_name, save_prefix)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def enchance(self, prompt_universe_size, candidate, error_evaluations, current_round, output_dir, enhancement_id):
        """
        Enhance prompt by applying critic and actor feedback iteratively.
        """
        self.prompt_universe_size = prompt_universe_size
        self.current_round = current_round
        enhanced_candidates = []
        token_usage_tracker = utils.TokenUsageDetails()
        token_usage_tracker.init_components(['critic', 'actor', 'overall'])
        self.logger.log("Starting generation process")
        evaluation_map = {e['id']: e for e in error_evaluations}
        output_dir = os.path.join(self.create_output_dir(output_dir, f"candidate_{candidate.id}"), f"enhancement_num_{enhancement_id}")
        try:
            # Apply structural feedback if within allowed rounds
            if self.current_round <= self.config.max_structural_rounds:
                self.logger.log("Running structural critic")
                structural_feedback, token_usage_structural_critic = self.run_structural_critic(candidate, output_dir)

                self.logger.log("Applying structural feedback via actor.")
                structural_candidates, token_usage_actor_structural = self.run_actor(candidate, output_dir, None, structural_feedback, "structural")
                token_usage_tracker.add_token_usage_for_component(token_usage_structural_critic, "critic")
                token_usage_tracker.add_token_usage_for_component(token_usage_actor_structural, "actor")
                candidate = structural_candidates[0] if structural_candidates else candidate
            
            # Run error critic and get feedback
            self.logger.log("Running error critic")
            error_feedback, token_usage_critic = self.run_error_critic(candidate, error_evaluations, output_dir)
            
            # Apply error feedback, generating multiple new prompts
            self.logger.log("Applying error feedback via actor.")
            new_candidates, token_usage_actor_error = self.run_actor(candidate, output_dir, evaluation_map, error_feedback, "error")
            enhanced_candidates.extend(new_candidates)
                                    
            # Track token usage
            token_usage_tracker.add_token_usage_for_component(token_usage_critic, "critic")
            token_usage_tracker.add_token_usage_for_component(token_usage_actor_error, "actor")

        except Exception as e:
            self.logger.log(f"Error during prompt enhancement: {str(e)}")
            self.logger.log(traceback.format_exc(), False)
            raise Exception("Error in enhancement process") from e
        
        return enhanced_candidates, token_usage_tracker
        
    def reseed(self, new_seed):
        """Reseed the random number generator for SCULPT operations."""
        self.random_seed = new_seed
        self.rng = np.random.default_rng(new_seed)
        
    def run_structural_critic(self, candidate, output_dir):
        """
        Runs the structural critic module to analyze and evaluate the candidate's prompt tree structure.
        The critic evaluates the structured prompt tree and generates feedback on both structural aspects
        and any detected errors. The feedback is parsed from JSON response and organized into collections.
        """
        token_usage_critic = utils.TokenUsage()
        try:
            self.logger.log("Critic: Running structural critic module iteration")
            prompt_tree_json = json.dumps(candidate.tree_to_json(), indent=4)

            # Construct the critic prompt
            critic_prompt = self.structural_critic_template.replace("{parsed_prompt}", prompt_tree_json)
            
            # Save the constructed critic prompt
            with open(output_dir + "_structural_critic_prompt.txt", 'w', encoding="utf-8") as pf:
                pf.write(critic_prompt)

            try:
                _, critic_response_str, token_usage_i = llm.infer(
                    critic_prompt,
                    max_tokens=self.config.max_tokens_critic,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    model=self.config.model,
                    provider=self.config.provider
                )
                token_usage_critic.update_token_usage(token_usage_i)
                critic_response_str = utils.clean_llm_response(critic_response_str)
            except Exception as e:
                self.logger.log(f"Error invoking LLM for  structural critic at iteration: {e}")
                self.logger.log(traceback.format_exc(), False)
                return [], token_usage_critic

            response_filename = output_dir + "_structural_critic_response.txt"
            with open(response_filename, 'w', encoding="utf-8") as rf:
                rf.write(critic_response_str)

            try:
                critic_feedback = dirtyjson.loads(critic_response_str)
                critic_feedback = [critic_feedback]
                
            except ValueError as e: 
                self.logger.log(f"Error while loading structural critic response {e} is not valid JSON.")
                return [], token_usage_critic

        except Exception as e:
            self.logger.log(f"Critical error in critic module: {e}")
            self.logger.log(traceback.format_exc(), False)
            raise Exception("Error in running critic") from e

        return critic_feedback, token_usage_critic
                
    def run_error_critic(self, candidate, error_evaluations, output_dir):
        """
        Runs the critic module, analyzing the structured prompt tree and batch evaluation results.
        Aggregates error feedback and merges structural feedback into a single entry.
        """
        all_error_feedback = []
        token_usage_critic = utils.TokenUsage()
        n_critic = self.config.n_critic

        try:
            for i in tqdm(range(n_critic), total=n_critic, desc='Critic'):
                self.logger.log(f"Critic: Running critic module iteration {i}")

                error_samples = list(self.rng.choice(error_evaluations, size=min(len(error_evaluations), self.config.errors_per_critic), replace=False)) if error_evaluations else []
                prompt_tree_json = json.dumps(candidate.tree_to_json(), indent=4)
                error_samples_str = json.dumps(error_samples, indent=4)

                # Construct the critic prompt
                critic_prompt = self.error_critic_template.replace("{parsed_prompt}", prompt_tree_json)
                critic_prompt = critic_prompt.replace("{batch_evaluation}", error_samples_str)
                
                # Save the constructed critic prompt
                with open(output_dir + f"_error_critic_prompt_{i}.txt", 'w', encoding="utf-8") as pf:
                    pf.write(critic_prompt)

                try:
                    _, critic_response_str, token_usage_i = llm.infer(
                        critic_prompt,
                        max_tokens=self.config.max_tokens_critic,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        model=self.config.model,
                        provider=self.config.provider
                    )
                    token_usage_critic.update_token_usage(token_usage_i)
                    critic_response_str = utils.clean_llm_response(critic_response_str)
                except Exception as e:
                    self.logger.log(f"Error invoking LLM for error critic at iteration {i}: {e}")
                    self.logger.log(traceback.format_exc(), False)
                    continue

                response_filename = output_dir + f"_error_critic_response_{i}.txt"
                with open(response_filename, 'w', encoding="utf-8") as rf:
                    rf.write(critic_response_str)

                try:
                    critic_feedback = dirtyjson.loads(critic_response_str)
                except ValueError as e:
                    self.logger.log(f"Error while loading critic response {e} at iteration {i} is not valid JSON. Skipping iteration {i}.")
                    continue
                
                if "error_feedback" in critic_feedback:
                    all_error_feedback.extend(critic_feedback["error_feedback"])

        except Exception as e:
            self.logger.log(f"Critical error in error critic module: {e}")
            self.logger.log(traceback.format_exc(), False)
            raise Exception("Error in running error critic") from e

        # Merge all structural feedbacks into a single entry
        aggregated_error_feedback = all_error_feedback
        if self.config.aggregate_feedbacks == "node_agg":
            aggregated_error_feedback = self.aggregate_by_shared_nodes(all_error_feedback)
        elif self.config.aggregate_feedbacks == "similarity_agg":
            aggregated_error_feedback, similarity_threshold = self.aggregate_by_similarity(all_error_feedback, self.config.min_similarity_threshold)   
            self.logger.log(f"Similarity threshold is {similarity_threshold}")
        elif self.config.aggregate_feedbacks == "diversity_agg":
            num_diverse_groups = max(1, int(len(all_error_feedback) ** self.config.aggregation_factor))
            aggregated_error_feedback = self.aggregate_by_diversity(all_error_feedback, num_diverse_groups)

        # Save the aggregated_error_feedback
        with open(output_dir + "_aggregated_error_feedback.json", 'w', encoding="utf-8") as f:
            json.dump(aggregated_error_feedback, f, indent=4)
        
        return aggregated_error_feedback, token_usage_critic
    
    def aggregate_by_shared_nodes(self, error_feedbacks):
        """
        Group records that share ≥1 feedback‑node id, preserving the original
        record structure exactly as it came in.

        Parameters
        ----------
        error_feedbacks : list[dict]
            Each dict has the same format as your input.

        Returns
        -------
        list[list[dict]]
            A list of groups; every inner list contains the *unaltered* records
            that belong together.
        """
        # ---------- union‑find setup -------------------------------------------
        parent = list(range(len(error_feedbacks)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pj] = pi
        # -----------------------------------------------------------------------

        # Build node‑id → indices map
        node_to_indices = defaultdict(list)
        for idx, rec in enumerate(error_feedbacks):
            for node_id in rec.get("feedback", {}):
                node_to_indices[node_id].append(idx)

        # Union every pair of indices that share a node‑id
        for idxs in node_to_indices.values():
            for i, j in itertools.combinations(idxs, 2):
                union(i, j)

        # Collect groups by their root parent
        groups = defaultdict(list)
        for idx, rec in enumerate(error_feedbacks):
            groups[find(idx)].append(rec)

        return list(groups.values())


    def aggregate_by_similarity(self, error_feedbacks, min_threshold: float = 0.80):
        """
        Cluster `error_feedbacks` by semantic similarity of their
        'evaluation_reasoning' field, using a *data‑driven* threshold.

        Parameters
        ----------
        min_threshold : float, optional
            Lower bound on the dynamic threshold (default = 0.80).
        """
        # 1) Embed the reasoning texts
        reasoning_texts = [fb["evaluation_reasoning"] for fb in error_feedbacks]
        embeddings = self.encoder.encode(reasoning_texts, convert_to_tensor=True)

        # 2) Derive a cut‑off from the batch itself
        similarity_threshold = utils.dynamic_similarity_threshold(
            embeddings, min_threshold=min_threshold
        )
        
        # 3) Union‑find style clustering (unchanged except we use the new threshold)
        clustered_indices = set()
        semantic_clusters = []
        total = len(error_feedbacks)

        for base_idx in range(total):
            if base_idx in clustered_indices:
                continue
            cluster = [error_feedbacks[base_idx]]
            clustered_indices.add(base_idx)

            for target_idx in range(base_idx + 1, total):
                if target_idx in clustered_indices:
                    continue
                sim = util.cos_sim(embeddings[base_idx], embeddings[target_idx]).item()
                if sim >= similarity_threshold:
                    cluster.append(error_feedbacks[target_idx])
                    clustered_indices.add(target_idx)

            semantic_clusters.append(cluster)

        # 4) Merge singletons into one group (same as before)
        singletons = [c for c in semantic_clusters if len(c) == 1]
        multi     = [c for c in semantic_clusters if len(c) > 1]
        if singletons:
            multi.append([fb for c in singletons for fb in c])

        return multi, similarity_threshold

    def aggregate_by_diversity(self, error_feedbacks, group_size=3):
        """
        Groups error_feedbacks to maximize semantic diversity within each group.

        This function selects error_feedbacks that are as semantically distinct as possible 
        from each other based on cosine similarity. Using a similarity matrix computed from 
        the embeddings of the 'reasoning' field, it iteratively chooses candidates with 
        the lowest average similarity to already selected feedbacks for each group.
        """
        
        # Extract reasoning texts from the error_feedbacks.
        reasoning_texts = [feedback["evaluation_reasoning"] for feedback in error_feedbacks]
        # Compute embeddings for the reasoning texts.
        embeddings = self.encoder.encode(reasoning_texts)
        total_feedbacks = len(error_feedbacks)

        # Compute a cosine similarity matrix for all embeddings.
        similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        np.fill_diagonal(similarity_matrix, 1.0)  # Ensure self-similarity is 1.

        # Set of indices for feedbacks that have not been assigned to a group.
        remaining_indices = set(range(total_feedbacks))
        diverse_groups = []

        # Iteratively form diverse groups while there are enough error_feedbacks left.
        while len(remaining_indices) >= group_size:
            group_indices = []
            candidate_indices = list(remaining_indices)

            # Choose the first available index as the seed for the new group.
            seed_idx = candidate_indices[0]
            group_indices.append(seed_idx)
            remaining_indices.remove(seed_idx)

            # Add additional feedbacks to the group by choosing those with minimum average similarity.
            while len(group_indices) < group_size:
                candidate_similarities = []
                for candidate_idx in list(remaining_indices):
                    avg_similarity = np.mean([similarity_matrix[candidate_idx][idx] for idx in group_indices])
                    candidate_similarities.append((avg_similarity, candidate_idx))
                candidate_similarities.sort(key=lambda x: x[0])
                _, chosen_idx = candidate_similarities[0]
                group_indices.append(chosen_idx)
                remaining_indices.remove(chosen_idx)

            diverse_groups.append([error_feedbacks[i] for i in group_indices])

        # If there are any remaining ungrouped feedbacks, create a final group with them.
        if remaining_indices:
            diverse_groups.append([error_feedbacks[i] for i in remaining_indices])

        return diverse_groups

    def extract_node_ids(self, feedbacks):
        """
        Extracts unique node IDs from a group of error feedback dictionaries.
        """
        node_ids = set()
        # check if feedback is list or dictionary
        if isinstance(feedbacks, dict):
            for node_id in feedbacks["feedback"]:
                node_ids.add(node_id)
        else:
            for feedback in feedbacks:
                for node_id in feedback["feedback"]:
                    node_ids.add(node_id)
        return list(node_ids)


    def run_actor(self, candidate, output_dir, evaluation_map, feedback_list, feedback_type="error"):
        """
        Applies the actor model to modify the prompt tree based on given feedbacks.
        """
        new_candidates = []
        token_usage_actor = utils.TokenUsage()
        
        for feedback_index, feedback in enumerate(tqdm(feedback_list, desc='Applying actor feedback')):
            try:
                self.logger.log(f"Actor: Running actor for feedback {feedback_index} in {output_dir}")
                evaluations_str = "None"
                
                # check whether the feedback is list or dict, if it is list then convert the list to dict then add NodesToUpdate
                if isinstance(feedback, list):
                    updated_feedback = {"AggregatedFeedback": feedback}
                    updated_feedback["NodesToUpdate"] = self.extract_node_ids(feedback)
                    if evaluation_map:
                        evaluations_str = json.dumps([evaluation_map[fb['id']] for fb in feedback if fb['id'] in evaluation_map], indent=4)
                        if len(evaluations_str.split()) > 500:
                            single_evaluation = json.dumps([evaluation_map[feedback[0]['id']] if feedback and feedback[0]['id'] in evaluation_map else []], indent=4)
                            evaluations_str = single_evaluation if len(single_evaluation.split()) <= 1000 else ""
                else:
                    updated_feedback = feedback
                    updated_feedback["NodesToUpdate"] = self.extract_node_ids(feedback)
                nodestoupdate = candidate.collect_node_and_subnodes(updated_feedback["NodesToUpdate"])

                if len(updated_feedback["NodesToUpdate"]) > len(updated_feedback["NodesToUpdate"]):
                    additional_nodes = [node for node in nodestoupdate if node not in updated_feedback["NodesToUpdate"]]
                    self.logger.log(f"Actor: Additional nodes added but not in original feedback: {additional_nodes}")
                
                updated_feedback["NodesToUpdate"] = nodestoupdate
                self.logger.log(f"Actor: Nodes to update: {updated_feedback['NodesToUpdate']}")                
                redacted_tree = candidate.redact_tree_for_nodes(updated_feedback["NodesToUpdate"])
                redacted_json = json.dumps(redacted_tree.tree_to_json(), indent=4)
                feedback_str = json.dumps(updated_feedback, indent=4)
                
                # Construct actor prompt
                actor_prompt = self.actor_template.replace("{parsed_prompt}", redacted_json)
                actor_prompt = actor_prompt.replace("{critic_feedback}", feedback_str)
                actor_prompt = actor_prompt.replace("{batch_evaluation}", evaluations_str)
                
                # Save the actor prompt
                with open(f"{output_dir}_{feedback_type}_actor_prompt_{feedback_index}.txt", 'w', encoding="utf-8") as f:
                    f.write(actor_prompt)
                
                # Call LLM for actor response
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
                
                # Save the actor response
                with open(f"{output_dir}_{feedback_type}_actor_response_{feedback_index}.txt", 'w', encoding="utf-8") as f:
                    f.write(actor_response_str)
                
                # Parse actions from actor response
                actions = dirtyjson.loads(actor_response_str)

                self.logger.log(f"Actor: Applying actions for feedback {feedback_index} in {output_dir}")
                
                # Apply actions to generate a new prompt
                new_candidate = self.apply_actions(candidate, actions["Actions"])
                self.logger.log(f"Actor: Generated new candidate for feedback {feedback_index} in {output_dir}")
                
                if feedback_type == "error":
                    self.logger.log(f"Actor: Updating prompt universe size from {self.prompt_universe_size}")
                    self.prompt_universe_size += 1
                    new_candidate.id = self.prompt_universe_size
                    new_candidate.parent_id = candidate.id
                    new_candidate.round_created = self.current_round
                
                # Save structured and formatted prompt
                with open(f"{output_dir}_{feedback_type}_new_structured_candidate_{new_candidate.id}.txt", 'w', encoding="utf-8") as f:
                    f.write(json.dumps(new_candidate.get_json_prompt(), indent=4))
                with open(f"{output_dir}_{feedback_type}_new_candidate_{new_candidate.id}.txt", 'w', encoding="utf-8") as f:
                    f.write(str(new_candidate.get_prompt_chat_template()))
                
                new_candidates.append(new_candidate)
            except Exception as e:
                self.logger.log(f"Error {e} while running actor for candidate {candidate.id} {feedback_index} in {output_dir}")
                self.logger.log(traceback.format_exc(), False)
        
        return new_candidates, token_usage_actor

    def apply_actions(self, candidate, actions):
        """
        Applies a list of actions to a copied PromptTree instance to generate an updated candidate.
        """
        updated_candidate = candidate.copy_and_reset()
        for action in actions:
            try:
                action_type = action["Action"]
                action = action["Params"]
                self.logger.log(f"Applying action: {action_type} on {action.get('NodeID', 'N/A')}")
    
                if action_type == "UPDATE_NODE_VALUE":
                    updated_candidate.update_node_value(action["NodeID"], action["NewValue"])
    
                elif action_type == "INSERT_NODE":
                    updated_candidate.insert_node(action["ParentID"], action["NewNode"])
    
                elif action_type == "DELETE_NODE":
                    updated_candidate.delete_node(action["NodeID"])
    
                elif action_type == "MERGE_NODES":
                    updated_candidate.merge_nodes(
                        action["ParentID"],
                        action["NodeID1"],
                        action["NodeID2"],
                        action["MergedContent"]
                    )
    
                elif action_type == "REORDER_NODE":
                    updated_candidate.reorder_node(action["NodeID"], action["NewOrder"])
    
                else:
                    self.logger.log(f"Unknown action type: {action_type}. Skipping.")
    
            except Exception as e:
                self.logger.log(f"Error applying action {action.get('Action', 'N/A')} on NodeID: {action.get('NodeID', 'N/A')} - {str(e)}")
                self.logger.log(traceback.format_exc(), False)
    
        return updated_candidate
    