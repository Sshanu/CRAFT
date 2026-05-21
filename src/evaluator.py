import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import utils
import os
import numpy as np
import concurrent.futures
from ucb import UCBBandits
from validation_subset_creator import ValidationSubsetCreator

class Evaluator:
    """Evaluator with different evaluation algorithms for validation and testing."""
    def __init__(self, config, logger, evaluation_manager, validation_set, output_dir=None, random_seed=42, score_weight=0.7, token_weight=0.3):
        self.config = config
        self.logger = logger
        self.evaluation_manager = evaluation_manager
        self.validation_set = validation_set
        self.output_dir = output_dir
        # Create seeded random generator for reproducible evaluation
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed
        self.score_weight = score_weight
        self.token_weight = token_weight
        
        if hasattr(self.config, 'validation_type'):
            assert self.config.validation_type in {'ucb', 'brute_force'}, f'Unknown validation type: {self.config.validation_type}'
        else:
            self.config.validation_type = 'ucb' #default
        
        if self.config.validation_type == 'ucb':
            self.validation = self.ucb_evaluation
        else:
            self.validation = self.bf_evaluation
            
            
        # Create validation subsets if enabled
        self.validation_subsets = []
        if self.config.use_stratified_validation:
            subset_creator = ValidationSubsetCreator(self.logger, self.config.embedding_model, self.config.eval_rounds)
            subset_creator = ValidationSubsetCreator(
                logger=self.logger,
                embedding_model=self.config.embedding_model,
                n_folds=self.config.eval_rounds,
                seed=random_seed,  # Use the passed random seed
                include_output_in_text_for_clustering=True,
                normalize_embeddings=True
            )
            self.validation_subsets = subset_creator.create_subsets(self.validation_set)
            
            # Save subsets to files if output directory is provided
            if self.output_dir:
                self._save_validation_subsets()
                
    def _save_validation_subsets(self):
        """
        Save validation subsets to separate JSON files.
        """
        import json
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            for i, subset in enumerate(self.validation_subsets):
                subset_file = os.path.join(self.output_dir, f"subset_{i}.json")
                with open(subset_file, 'w', encoding='utf-8') as f:
                    json.dump(subset, f, indent=2, ensure_ascii=False)
                self.logger.log(f"Saved validation subset {i} with {len(subset)} examples to {subset_file}")
            
            # Save metadata
            metadata = {
                'n_folds': self.config.eval_rounds,
                'total_examples': sum(len(subset) for subset in self.validation_subsets),
                'examples_per_subset': [len(subset) for subset in self.validation_subsets],
                'embedding_model': self.config.embedding_model
            }
            
            metadata_file = os.path.join(self.output_dir, "metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.log(f"Saved validation subsets metadata to {metadata_file}")
            
        except Exception as e:
            self.logger.log(f"Error saving validation subsets: {e}")
            
    def ucb_evaluation(self, prompt_candidates, output_dir, evaluation_mode='Validation'):
        """Evaluates prompt candidates using the UCB bandit algorithm."""
        sampled_history = set()

        def data_sampler(samples):
            nonlocal sampled_history
            available_samples = [s for s in samples if s["id"] not in sampled_history]
            if len(available_samples) <= self.config.samples_per_eval:
                sampled = available_samples
            else:
                # Use seeded random generator for reproducible sampling
                indices = self.rng.choice(len(available_samples), size=self.config.samples_per_eval, replace=False)
                sampled = [available_samples[i] for i in indices]
            sampled_history.update(s["id"] for s in sampled)
            return sampled

        # Calculate token lengths for each prompt candidate.
        token_lengths = []
        for prompt in prompt_candidates:
            if prompt.token_length is None:
                prompt.calculate_token_length()
            token_lengths.append(prompt.token_length)
            
        bandit_algo = UCBBandits(
            self.logger, 
            len(prompt_candidates), 
            token_lengths, 
            num_samples=self.config.samples_per_eval, 
            mode=self.config.ucb_type, 
            c=self.config.c,
            random_seed=self.random_seed,
            score_weight=self.score_weight,
            token_weight=self.token_weight
        )
        num_prompts_per_round = min(self.config.eval_prompts_per_round, len(prompt_candidates))
        token_usage = utils.TokenUsage()
        previous_sampled_indices = None
        previous_previous_sampled_indices = None

        for ri in tqdm(range(self.config.eval_rounds), desc=f'Evaluating {len(prompt_candidates)} prompts'):
            sampled_indices = bandit_algo.choose(num_prompts_per_round, ri, max_t=self.config.eval_rounds)

            # Check for early stopping if the same candidates are selected for 2 consecutive steps
            if (previous_sampled_indices is not None and 
                previous_previous_sampled_indices is not None and 
                set(sampled_indices) == set(previous_sampled_indices) and 
                set(previous_sampled_indices) == set(previous_previous_sampled_indices)):
                self.logger.log(f"Early stopping at round {ri}: Same candidates selected for 2 consecutive rounds")
                break

            previous_previous_sampled_indices = previous_sampled_indices
            previous_sampled_indices = sampled_indices
            sampled_prompts = [prompt_candidates[i] for i in sampled_indices]
            
            if self.config.use_stratified_validation:
                sampled_data = self.validation_subsets[ri]
            else:
                sampled_data = data_sampler(self.validation_set)
                
            scores = []

            # Evaluate the sampled prompts.
            max_workers = min(2, int(math.sqrt(self.config.max_threads)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for prompt in sampled_prompts:
                    filename = os.path.join(output_dir, f"{prompt.id}_validation_predictions_{ri}.tsv")
                # score, _, _, token_usage_i, _ = self.evaluation_manager.evaluate(prompt, sampled_data, filename, desc=f"{evaluation_mode} {prompt.id}")
                    futures[prompt] = executor.submit(self.evaluation_manager.evaluate, prompt, sampled_data, filename, desc=f"{evaluation_mode} {prompt.id}")
                
                for prompt in sampled_prompts:
                    try:
                        score, _, _, _, _ = futures[prompt].result()
                    except Exception as e:
                        self.logger.log(f"Error evaluating prompt {prompt.id}: {e}")
                        score = 0
                    scores.append(score)
                    
            bandit_algo.update(sampled_indices, scores)
        ucb_scores = bandit_algo.get_scores().tolist()
        ucb_stds = bandit_algo.get_std().tolist()
        candidate_scores = []
        for i, prompt in enumerate(prompt_candidates):
            candidate_scores.append((prompt, ucb_scores[i], ucb_stds[i]))
        return candidate_scores, token_usage
        
    def reseed(self, new_seed):
        """Reseed the random number generator for this evaluator."""
        self.random_seed = new_seed
        self.rng = np.random.default_rng(new_seed)

    def bf_evaluation(self, prompt_candidates, output_dir, evaluation_mode='Validation'):
        """Tests prompt candidates using the brute force algorithm."""
        token_usage = utils.TokenUsage()
        candidate_scores = []
        for prompt in tqdm(prompt_candidates, desc=f'Brute force Evaluating {len(prompt_candidates)} prompts'):
            filename = os.path.join(output_dir, f"{prompt.id}_{evaluation_mode}_predictions.tsv")
            if prompt.token_length is None:
                prompt.calculate_token_length()
            score, _, _, token_usage_i, _ = self.evaluation_manager.evaluate(
                prompt, self.validation_set, filename, desc=f"{evaluation_mode} {prompt.id}")
            token_usage.update_token_usage(token_usage_i)
            candidate_scores.append((prompt, score, 0.0))  # Default eval_std to 0.0 for brute force
        return candidate_scores, token_usage