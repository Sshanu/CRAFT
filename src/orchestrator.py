from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import math
import random
import json
import numpy as np
from tqdm import tqdm
import prompt_template
import utils
from fewshot_prompt_generator import FewShotPromptGenerator
from evaluator import Evaluator
from sculpt import Sculpt
from opro import OPRO
from evoprompt import EvoPrompt
from candidate_selector import CandidateSelector
from prompt_graph import PromptExpansionGraph
from prompt_compressor import PromptCompressor
from pareto_metrics import ParetoMetrics
from mutator import PromptMutator

class Orchestrator:
    """Orchestrator for managing the prompt generation and evaluation process."""
    def __init__(self, task_name, task, eval_manager, config, project_root, output_dir, logger, wandb_logger=None, random_seed=42):
        """
        Initialize the Orchestrator.
        """
        self.task_name = task_name
        self.task = task
        self.config = config.orchestrator
        self.logger = logger
        self.wandb_logger = wandb_logger
        self.random_seed = random_seed
        
        # Ensure task has proper random seed
        if hasattr(self.task, 'reseed'):
            self.task.reseed(random_seed)
        elif hasattr(self.task, 'rng'):
            self.task.rng = np.random.default_rng(random_seed)
            
        # Create seeded random generator for orchestrator operations
        self.rng = np.random.default_rng(random_seed)

        # Load data
        self.train_examples = self.task.get_train_examples()
        self.val_examples = self.sample_data(self.task.get_validation_examples(), self.config.n_val_exs)
        self.test_examples = self.sample_data(self.task.get_test_examples(), self.config.n_test_exs)
        self.logger.log(f"Train examples: {len(self.train_examples)}, Validation examples: {len(self.val_examples)}, Test examples: {len(self.test_examples)}")

        # Adjust minibatch size
        if self.config.minibatch_size >= len(self.train_examples):
            self.config.minibatch_size = len(self.train_examples) // 2

        # Initialize enhancer
        if config.enhancer.type == "sculpt":
            self.enhancer = Sculpt(config.enhancer, logger, random_seed=random_seed)
        elif config.enhancer.type == "opro":
            self.enhancer = OPRO(config.enhancer, logger, random_seed=random_seed)
        elif config.enhancer.type == "evoprompt":
            self.enhancer = EvoPrompt(config.enhancer, logger, random_seed=random_seed)
        else:
            raise ValueError(f"Unknown enhancer type: {config.enhancer.type}")
        self.enhancer.max_rounds = config.max_rounds

        # Selector and graph
        self.selector = CandidateSelector(config.selector, logger, random_seed=random_seed)
        self.expansion_graph = PromptExpansionGraph(logger)

        # Evaluator setup
        if len(self.val_examples) < 80:
            config.evaluator.eval_rounds = int(config.evaluator.eval_rounds/2)
            
        config.evaluator.samples_per_eval = int(
            min(
                max(len(self.val_examples) // 2, 80),
                len(self.val_examples)
            ) / config.evaluator.eval_rounds
        )
        config.evaluator.eval_prompts_per_round = 2 * config.selector.beam_size
        self.logger.log(f"Samples per eval: {config.evaluator.samples_per_eval}")

        # Prompt modules
        self.initial_prompt_generator = FewShotPromptGenerator(self.config, logger)
        self.mutator = PromptMutator(config.mutator, logger)

        # Create two ParetoMetrics instances: one for eval_score and one for test_score.
        self.pareto_metrics_eval = ParetoMetrics(
            os.path.join(output_dir, "pareto_metrics_eval"),
            wandb_logger=self.wandb_logger,
            random_seed=random_seed
        )
        self.pareto_metrics_test = ParetoMetrics(
            os.path.join(output_dir, "pareto_metrics_test"),
            wandb_logger=self.wandb_logger,
            random_seed=random_seed
        )

        self.eval_manager = eval_manager
        self.project_root = project_root
        self.output_dir = output_dir

        # Directories
        self.gen_prompts_dir = os.path.join(output_dir, "generated_prompts")
        self.temp_dir = os.path.join(output_dir, "temp")
        self.val_subsets_dir = os.path.join(output_dir, "validation_subsets")
        os.makedirs(self.gen_prompts_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.val_subsets_dir, exist_ok=True)
        
        self.prompt_compressor = PromptCompressor(config.compressor, logger, self.config, self.temp_dir)
        self.evaluator = Evaluator(config.evaluator, logger, eval_manager, self.val_examples, self.val_subsets_dir, random_seed=random_seed, score_weight=config.selector.score_weight, token_weight=config.selector.token_weight)

        # ID counter and state
        self.prompt_universe_size = 0
        self.top_candidates = []
        self.initial_prompts = []
        self.eval_round_numbers = []

    def update_prompt_universe_size(self):
        """Updates the prompt universe size that will be used as prompt id."""
        self.prompt_universe_size += 1
        
    def _run_parallel(self, jobs, worker_fn, desc: str, max_workers: int,
                       max_failure_ratio: float = 0.5,
                       job_retries: int = 1):
        """Execute jobs in parallel using ThreadPoolExecutor.

        Collects non-None results in submission order (deterministic).

        - Each failed job is retried up to `job_retries` extra times (default 1).
          Most production failures are transient (rate-limit blips, momentary
          network errors), so one retry catches the bulk of them without
          significantly compounding cost. Exceptions still bubble up if they
          persist across retries.
        - If too many jobs ultimately fail (> max_failure_ratio), raise
          instead of silently proceeding with degraded data. Prior behaviour
          was to swallow every exception, making silent candidate loss
          invisible at the call site.
        """
        ordered_results = [None] * len(jobs)
        failures: list[tuple[int, Exception]] = []
        retried_indices: set[int] = set()

        def _try_run(idx, job):
            return worker_fn(job)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_try_run, idx, job): idx
                             for idx, job in enumerate(jobs)}
            pbar = tqdm(total=len(jobs), desc=desc)
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    if idx not in retried_indices and job_retries > 0:
                        # Schedule an in-place retry for this job; don't tick
                        # the progress bar yet — we'll count the retry result.
                        self.logger.log(
                            f"[_run_parallel] Job {idx} in {desc!r} failed "
                            f"({e!r}); retrying once.")
                        retried_indices.add(idx)
                        retry_fut = executor.submit(_try_run, idx, jobs[idx])
                        future_to_idx[retry_fut] = idx
                        continue
                    self.logger.log(
                        f"[_run_parallel] Job {idx} in {desc!r} failed (final): {e!r}")
                    failures.append((idx, e))
                    res = None
                ordered_results[idx] = res
                pbar.update(1)
            pbar.close()

        if failures and len(failures) > max_failure_ratio * max(1, len(jobs)):
            self.logger.log(
                f"[_run_parallel] {len(failures)}/{len(jobs)} jobs failed in "
                f"{desc!r} (> {int(max_failure_ratio*100)}%). Aborting round.")
            first_idx, first_exc = failures[0]
            raise RuntimeError(
                f"_run_parallel: {len(failures)}/{len(jobs)} failures in {desc!r} "
                f"(threshold {max_failure_ratio:.0%}). First failure idx={first_idx}: "
                f"{first_exc!r}"
            ) from first_exc

        return [r for r in ordered_results if r is not None]

    def _integrate_trees(self, trees, metadata=""):
        """
        Add new PromptTree nodes to the expansion graph and save them.
        """
        for tree in trees:
            self.expansion_graph.add_node(tree)
        if trees:
            self.save_candidates(trees, self.gen_prompts_dir, metadata)            

    def _build_trees(self, items, desc: str, round_created: int = 0):
        """
        Parallel build of PromptTree objects given precomputed tree IDs.

        Args:
            items: List of tuples (tree_id, text, parent_id).
            desc: Progress description.
            round_created: round index for the trees.
        Returns:
            List of PromptTree.
        """
        # jobs are already tree_id, text, parent_id
        
        self.logger.log(f"Building {len(items)} PromptTree objects")
        def _builder(job):
            tree_id, text, parent_id = job
            return prompt_template.PromptTree(
                tree_id,
                text,
                self.config.model,
                self.config.provider,
                self.temp_dir,
                parent_id=parent_id,
                round_created=round_created,
                formatting=self.config.format_prompt,
            )

        results = self._run_parallel(items, _builder, desc=desc, max_workers=self.config.max_threads)
        # update universe size to max assigned ID
        max_id = max(tree_id for tree_id, _, _ in items) if items else self.prompt_universe_size
        self.prompt_universe_size = max(self.prompt_universe_size, max_id)
        return results
        
    def sample_train_batch(self, sample_size):
        """
        Samples a batch of training examples.
        """
        return self.task.data_sampler(self.train_examples, sample_size, mode=self.config.sample_type)
    
    def sample_data(self, examples, size):
        """
        Samples a subset of data from the provided examples.
        """
        if size >= len(examples):
            return examples
        return self.task.data_sampler(examples, size, mode=self.config.sample_type)
    
    def save_candidates(self, candidates, save_dir, metadata=""):
        """
        Save the generated candidates to the specified directory.
        """
        os.makedirs(save_dir, exist_ok=True)
        for candidate in candidates:
            candidate_filepath = os.path.join(save_dir, f"candidate_{candidate.id}_{metadata}.txt")
            with open(candidate_filepath, 'w', encoding='utf-8') as outf:
                outf.write(candidate.get_prompt_chat_template())

            candidate_filepath = os.path.join(save_dir, f"structured_candidate_{candidate.id}_{metadata}.txt")
            with open(candidate_filepath, 'w', encoding='utf-8') as outf:
                outf.write(json.dumps(candidate.get_json_prompt(), indent=4))
       
    def read_initial_prompts(self, initial_prompt_names):
        """Reads initial prompts from files and stores them in self.initial_prompts."""
        prompts_dir = os.path.join(self.project_root, "scenario", self.task_name, "prompts")
        for prompt_name in initial_prompt_names:
            file_path = os.path.join(prompts_dir, prompt_name)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    prompt_content = f.read()
                self.initial_prompts.append(prompt_content)
            else:
                self.logger.log(f"Warning: Prompt file '{file_path}' not found.")

    def build_few_shot_examples(self):
        """
        Samples and formats few-shot examples into a string.

        Returns:
            str: Formatted string of few-shot examples.
        """
        examples = self.sample_train_batch(self.config.num_fewshots)
        formatted_examples = []
        for ex in examples:
            formatted_examples.append(f"Input: {ex['input']}\nOutput: {ex['output']}")
        return "\n\n".join(formatted_examples)
    
    def structure_initial_prompts(self, current_round=0):
        """
        Convert raw initial prompt strings into PromptTree objects in parallel, then deduplicate.
        """
        # Remove placeholder candidates for raw prompts
        raw_count = len(self.initial_prompts)
        if raw_count:
            self.top_candidates = self.top_candidates[:-raw_count]

        # Snapshot existing candidates
        existing_candidates = list(self.top_candidates)

        # Prepare parallel build jobs with assigned IDs
        base_id = self.prompt_universe_size
        prompt_texts = self.initial_prompts  # List of raw prompt strings
        build_jobs = [
            (base_id + idx + 1, text, None)
            for idx, text in enumerate(prompt_texts)
        ]

        # Build PromptTree objects in parallel
        new_trees = self._build_trees(
            items=build_jobs,
            desc="Structuring initial prompts",
            round_created=current_round
        )

        # Integrate new trees and remove duplicates
        self._integrate_trees(new_trees, "initial")
        self.top_candidates = self.deduplicate_candidates(existing_candidates, new_trees)
        
    def generate_initial_prompts(
        self,
        output_dir: str,
    ):
        """
        Generate few-shot prompts in parallel, build PromptTrees, deduplicate.
        """
        if not self.initial_prompts:
            raise ValueError("initial_prompts list is empty")

        # Extract the user message segment
        first_prompt = self.initial_prompts[0]
        segments = prompt_template.extract_prompt_segments(first_prompt)
        user_message = segments.get("UserMessage")
        if not user_message:
            raise KeyError("'UserMessage' segment missing from first initial prompt")

        # Snapshot current candidates for deduplication
        previous_candidates = list(self.top_candidates)

        # Prepare generation jobs
        prompt_types = ["short", "medium", "long"]
        num_prompts = self.config.n_initial_prompts
        jobs = [(ptype, idx) for ptype in prompt_types for idx in range(num_prompts)]

        def _generate_job(job):
            ptype, idx = job
            examples = self.build_few_shot_examples()
            generated, _ = self.initial_prompt_generator.generate_prompt(
                user_message,
                ptype,
                examples,
                os.path.join(output_dir, f"{idx}_"),
            )
            if not generated or generated == "Could not get a response":
                return None
            return generated

        # Run generation in parallel
        raw_results = self._run_parallel(
            jobs,
            _generate_job,
            desc="Generating few-shot prompts",
            max_workers=self.config.max_threads,
        )
        # Filter out failed generations
        raw_results = [text for text in raw_results if text]

        # Assign IDs and prepare tree-building items
        start_id = self.prompt_universe_size
        tree_items = [
            (start_id + i + 1, text, None)
            for i, text in enumerate(raw_results)
        ]

        # Build PromptTree objects in parallel
        new_trees = self._build_trees(
            items=tree_items,
            desc="Building PromptTrees",
            round_created=-1,
        )

        # Integrate into graph and save
        self._integrate_trees(new_trees, "initial")

        # Deduplicate and update top_candidates
        self.top_candidates = self.deduplicate_candidates(previous_candidates, new_trees)

    def validate_candidates(self, output_dir, current_round=-1):
        """
        Validates the top candidates against the validation set.
        """
        token_usage = utils.TokenUsage()
        if current_round == 0:
            candidate_scores, token_usage = self.evaluator.bf_evaluation(
                self.top_candidates, output_dir, evaluation_mode='Validating')
        else:
            candidate_scores, token_usage = self.evaluator.validation(
                self.top_candidates, output_dir, evaluation_mode='Validating')
                
        # Create a validation report file.
        updated_candidates = []
        report_file = os.path.join(output_dir, "validation_report.tsv")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("prompt_id\tparent_prompt_id\teval_score\teval_std\ttoken_length\n")
            for candidate, score, std in candidate_scores:
                candidate.eval_score = score
                candidate.eval_std = round(std, 3)
                updated_candidates.append(candidate)
                parent_id = candidate.parent_id if candidate.parent_id is not None else ""
                f.write(f"{candidate.id}\t{parent_id}\t{candidate.eval_score}\t{candidate.eval_std}\t{candidate.token_length}\n")

        self.top_candidates = updated_candidates
        
        # Log validation metrics to wandb
        if self.wandb_logger:
            # Determine the current round for logging
            current_round = getattr(self, '_current_round', 0)
            self.wandb_logger.log_validation_metrics(current_round, updated_candidates, report_file)
        
        return token_usage

    def evaluate_candidates(self, output_dir):
        """
        Evaluates the top candidates on the test set.
        """
        self.logger.log("Evaluating candidates on test set")
        token_usage = utils.TokenUsage()
        test_reports  = []
        updated_candidate_list = []
        for candidate in self.top_candidates:
            test_filepath = os.path.join(output_dir, f"{candidate.id}_test_predictions.tsv")
            test_score, _, test_report, token_usage_i, _ = self.eval_manager.evaluate(
                candidate, self.test_examples, test_filepath,
                desc=f"Testing {candidate.id}"
            )
            candidate.test_score = test_score
            token_usage.update_token_usage(token_usage_i)
            test_reports.append(test_report)
            updated_candidate_list.append(candidate)
            
            json_filepath = os.path.join(output_dir, f"{candidate.id}_test_report.json")
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(test_report, f, indent=4)
        
        self.top_candidates = updated_candidate_list
        
        # Log test evaluation metrics to wandb
        if self.wandb_logger:
            current_round = getattr(self, '_current_round', 0)
            test_report_path = os.path.join(output_dir, "test_report.tsv")
            self.wandb_logger.log_test_metrics(current_round, updated_candidate_list, test_reports, test_report_path)
        
        return test_reports, token_usage

    def evaluate_and_analyze_errors(self, candidate, train_batch, round_dir):
        """
        Evaluates the prompt candidate on the train_batch and analyzes errors.
        """
        _, output_exs, _, token_usage_eval, _ = self.eval_manager.evaluate(
            candidate,
            train_batch,
            os.path.join(round_dir, f"candidate_{candidate.id}_train_batch.tsv"),
            desc=f"Evaluate train batch using {candidate.id}"
        )
        error_evaluations = self.eval_manager.get_error_batch(output_exs)
        # Stable sort by example id so downstream Sculpt sampling
        # (np.random.choice with seeded RNG) sees a deterministic list order.
        # Without this, even with identical seeds the critic samples different
        # errors between runs, causing prompt-tree divergence and cache misses.
        error_evaluations = sorted(error_evaluations, key=lambda e: str(e.get('id', '')))
        return error_evaluations, token_usage_eval

    def enhance_single_candidate(self, candidate, error_evaluations, current_round, round_dir):
        """
        Enhance a single candidate by generating new candidates.
        """
        new_candidates = []
        token_usage_expansion = utils.TokenUsageDetails()
        token_usage_expansion.init_components(['enhancer'])

        # Provide history (top_candidates validated so far) to enhancers that
        # consume it. Sculpt ignores this attribute; OPRO uses it as fewshot
        # history (top-K by eval_score).
        self.enhancer.population_candidates = list(self.top_candidates)

        for i in tqdm(range(self.config.n_enchanement), total=self.config.n_enchanement, desc='Enhancing..'):
            self.logger.log(f"Enhancing Candidate {candidate.id}, enhancement no. {i}")
            # Create a fresh copy of the candidate for each enhancement iteration
            candidate_copy = candidate.create_candidate_copy()
            # Generate new candidates using enhancer
            new_expansions, cur_token_usage = self.enhancer.enchance(self.prompt_universe_size, candidate_copy, error_evaluations, current_round, round_dir, i) # error_evaluations is not used here
            self.prompt_universe_size = self.enhancer.prompt_universe_size
            token_usage_expansion.add_token_usage_for_component(cur_token_usage, "expansion", add=False)
            new_candidates += new_expansions

            # Save new enhanced candidate texts.
            self.save_candidates(new_expansions, self.gen_prompts_dir, metadata="generated")

        return new_candidates, token_usage_expansion

    def filter_candidates(self, new_candidates, error_evaluations, round_dir, candidate):
        """
        Filters the new candidates based on the reject_on_errors configuration.
        """
        token_usage_eval = utils.TokenUsageDetails()
        token_usage_eval.init_components(['eval'])

        if len(new_candidates) > self.config.max_expansion_factor:
            if self.config.reject_on_errors:
                # Sample error examples using evaluator with seeded random generator.
                sampled_errors = list(self.rng.choice(error_evaluations, size=min(len(error_evaluations), self.config.roe_batch_size), replace=False)) if error_evaluations else []

                # Reduce candidate list by sampling up to twice the max expansion factor.
                candidates_to_sample = min(len(new_candidates), self.config.max_expansion_factor * 2)
                sampled_indices = self.rng.choice(len(new_candidates), size=candidates_to_sample, replace=False)
                new_candidates = [new_candidates[i] for i in sampled_indices]
                os.path.join(round_dir + f"candidate_{candidate.id}")
                report_filepath = os.path.join(round_dir, f"candidate_{candidate.id}_rejectonerror_report.tsv")
                error_scores = []
                with open(report_filepath, "w", encoding="utf-8") as rf:
                    rf.write("prompt_id\tparent_prompt_id\tscore\n")
                    for _, new_candidate in enumerate(new_candidates):
                        error_score, _, _, tokens, _ = self.eval_manager.evaluate(
                            new_candidate,
                            sampled_errors,
                            os.path.join(round_dir,  f"{new_candidate.id}_rejectonerror_predictions.tsv"),
                            desc=f"Evaluate RejectOnError using {new_candidate.id}"
                        )
                        token_usage_eval.add_token_usage_for_component(tokens, "eval")
                        error_scores.append(error_score)
                        parent_id = new_candidate.parent_id if new_candidate.parent_id is not None else ""
                        rf.write(f"{new_candidate.id}\t{parent_id}\t{error_score}\n")
                
                # Select top candidates based on error score.
                sorted_indices = np.argsort(error_scores)[-self.config.max_expansion_factor:]
                new_candidates = [new_candidates[i] for i in sorted_indices]
            else:
                # Sample candidates using seeded random generator
                sample_indices = self.rng.choice(len(new_candidates), size=self.config.max_expansion_factor, replace=False)
                new_candidates = [new_candidates[i] for i in sample_indices]
        
        return new_candidates, token_usage_eval

    def deduplicate_candidates(self, existing_candidates, new_candidates):
        """
        Deduplicates the candidates.
        """
        seen_candidates = set()
        deduplicated_candidates = []
        overall_candidates = existing_candidates + new_candidates
        for obj in overall_candidates:
            # Print the type of obj
            if not isinstance(obj, prompt_template.PromptTree):
                self.logger.log(f"Warning: Object {obj} is not a PromptTree, type: {type(obj)}")
                continue
            if obj.get_prompt_chat_template() not in seen_candidates:
                deduplicated_candidates.append(obj)
                seen_candidates.add(obj.get_prompt_chat_template())
            else:
                self.logger.log(f"Deduplicated candidate: {obj.id} with text: {obj.get_prompt_chat_template()}")
        return deduplicated_candidates

    def enhance_candidates(self, train_batch, current_round: int, round_dir: str):
        """
        Enhance the candidates by generating new candidates and filtering them.
        """
        self.logger.log(f"Enhancing {len(self.top_candidates)} candidates with {self.config.n_enchanement} enhancements each")
        # Snapshot current candidates
        previous_candidates = self.top_candidates
        
        # Worker: evaluate, enhance, filter a single candidate
        def _expand_worker(candidate):
            # Evaluate and analyze errors
            errors, _ = self.evaluate_and_analyze_errors(candidate, train_batch, round_dir)
            if not errors:
                return []
            # Enhance single candidate
            new_cands, _ = self.enhance_single_candidate(candidate, errors, current_round, round_dir)
            if not new_cands:
                return []
            # Filter enhanced candidates
            filtered, _ = self.filter_candidates(new_cands, errors, round_dir, candidate)
            return filtered

        # Run expansion in parallel
        jobs = previous_candidates
        results = self._run_parallel(
            jobs,
            _expand_worker,
            desc="Enhancing candidates",
            max_workers=self.config.max_threads,
        )
        all_new_trees = [tree for sublist in results for tree in sublist if tree is not None]        
        return all_new_trees
    
    def mutate_candidates(self, current_round: int, round_dir: str):
        """
        Mutates each candidate self.config.num_mutations times in parallel,
        builds trees, and deduplicates.
        """
        num_mutations = self.mutator.config.num_mutations
        self.logger.log(f"Mutating {len(self.top_candidates)} candidates × {num_mutations} times")

        token_usage_mutation = utils.TokenUsageDetails()
        token_usage_mutation.init_components(['mutation'])

        # Create multiple jobs per candidate
        mutation_jobs = []
        for cand in list(self.top_candidates):
            for m_idx in range(num_mutations):
                save_prefix = f"mutation_{m_idx}"
                mutation_jobs.append((cand, save_prefix))

        def mutate_worker(job):
            cand, save_prefix = job
            # Create a fresh copy of the candidate for mutation
            candidate_copy = cand.create_candidate_copy()
            task_examples_str = self.build_few_shot_examples()
            mutated_prompt_text, _ = self.mutator.mutate_prompt(
                candidate_copy, save_prefix, task_examples_str, round_dir
            )

            if mutated_prompt_text:
                return (mutated_prompt_text, cand.id)
            return None

        # Run mutation jobs in parallel
        mutation_results = self._run_parallel(
            mutation_jobs,
            mutate_worker,
            desc=f"Mutating candidates (×{num_mutations})",
            max_workers=self.config.max_threads,
        )

        # Filter out failed mutations
        mutation_results = [res for res in mutation_results if res]

        # Assign new IDs and prepare tree data
        start_id = self.prompt_universe_size
        tree_data = [
            (start_id + idx + 1, mutated_text, parent_id)
            for idx, (mutated_text, parent_id) in enumerate(mutation_results)
        ]
        self.prompt_universe_size += len(tree_data)

        # Build PromptTrees
        new_mutated_trees = self._build_trees(
            tree_data,
            desc="Building Mutated PromptTrees",
            round_created=current_round,
        )

        # Integrate and deduplicate
        self._integrate_trees(new_mutated_trees, "mutated")
        return new_mutated_trees

    def compress_candidates(self, train_batch, current_round: int, round_dir: str):
        """
        Compresses the candidates using various strategies.
        """
        self.logger.log(f"Compressing {len(self.top_candidates)} candidates")
        previous_candidates = list(self.top_candidates)
        
        # Generate compress jobs
        ratios = np.round(
            np.linspace(
                self.prompt_compressor.config.min_ratio,
                self.prompt_compressor.config.max_ratio,
                self.prompt_compressor.config.num_compressions,
            ),
            2,
        )

        def compress_worker(job):
            candidate, ratio, new_prompt_id_start = job
            # Create a fresh copy of the candidate for compression
            candidate_copy = candidate.create_candidate_copy()
            return self.prompt_compressor.compress(
                candidate_copy,
                train_batch,
                ratio,
                current_round,
                round_dir,
                new_prompt_id_start
            )

        num_strategies = len(self.prompt_compressor.strategies)
        jobs = []
        i = 0
        for candidate in previous_candidates:
            for ratio in ratios:
                new_prompt_id_start = self.prompt_universe_size + i * num_strategies + 1
                jobs.append((candidate, ratio, new_prompt_id_start))
                i += 1
        
        if not jobs:
            return []
        
        self.prompt_universe_size += i * num_strategies

        compressed_results = self._run_parallel(
            jobs,
            compress_worker,
            desc="Compressing candidates",
            max_workers=self.config.max_threads,
        )
        
        # Process and integrate the compressed trees
        all_new_trees = [tree for sublist in compressed_results for tree in sublist if tree is not None]
        self.logger.log(f"Generated {len(all_new_trees)} new compressed candidates")
        self._integrate_trees(all_new_trees, "compressed")
        return all_new_trees

    def process_candidates(self, current_round_dir, current_round, run_test_evals=False):
        """
        Process the candidates by validating, selecting, and evaluating them.
        """
        # Store current round for wandb logging
        self._current_round = current_round
        if self.enhancer.max_rounds>0:
            # Validate candidates
            self.logger.log(f"Validating {len(self.top_candidates)} candidates in round {current_round}")
            validation_results = self.validate_candidates(current_round_dir)
            self.logger.log(f"Validation complete at round {current_round}")

            # Filter bad candidates
            min_points = int(math.ceil(self.selector.config.beam_size * 1.5))
            if len(self.top_candidates) > min_points:
                self.logger.log(f"Filtering bad candidates at round {current_round}")
                progression_ratio = current_round / self.enhancer.max_rounds if self.enhancer.max_rounds > 0 else 1.0
                filtered_candidates = self.selector.filter_bad_candidates(
                    self.top_candidates, 
                    score_attr="eval_score",
                    std_attr="eval_std",
                    filter_factor=self.selector.config.filter_factor,
                    progression=progression_ratio,
                    min_points=min_points
                )
                self.top_candidates = filtered_candidates
            else:
                self.logger.log(f"Skipping filtering as candidate count {len(self.top_candidates)} <= beam size {self.selector.config.beam_size}")

            # Select candidates
            self.logger.log(f"Selecting candidates at round {current_round}")
            selected_candidates = self.selector.select_candidates(self.top_candidates, current_round/self.enhancer.max_rounds)
            self.top_candidates = selected_candidates

            # Log selected candidates
            candidate_info_str = ", ".join(
                [f"{cand.id} (eval: {cand.eval_score}, tokens: {cand.token_length})"
                for cand in selected_candidates]
            )
            self.logger.log(f"Scoring complete: {candidate_info_str} at round {current_round}")
            
            # Log selected candidates to wandb
            if self.wandb_logger:
                self.wandb_logger.log_selected_candidates(current_round, selected_candidates)

            # Save candidate prompt texts
            self.logger.log("Saving candidate prompts")
            for i, candidate in enumerate(selected_candidates):
                candidate_filepath = os.path.join(current_round_dir, f"candidate_{candidate.id}_rank_{i}.txt")
                with open(candidate_filepath, 'w', encoding='utf-8') as outf:
                    outf.write(candidate.get_prompt_chat_template())

        # Run test evaluations if applicable
        if run_test_evals:
            self.logger.log(f"Running test evaluations for round {current_round}")
            self.eval_round_numbers.append(current_round)
            test_reports, _ = self.evaluate_candidates(current_round_dir)
            merged_test_report = self.eval_manager.merge_reports(self.top_candidates, test_reports)
            merged_test_report.to_csv(os.path.join(current_round_dir, "test_report.tsv"), sep='\t', index=False)
            roundwise_report = self.eval_manager.aggregate_report_across_rounds(merged_test_report, current_round)
            roundwise_report.to_csv(os.path.join(self.output_dir, "test_report.tsv"), sep='\t', index=False)

            # Calculate and plot Pareto metrics
            self.logger.log(f"Calculating Pareto metrics for round {current_round}")
            self.pareto_metrics_test.calculate(current_round, self.top_candidates, score_attr='test_score', token_attr='token_length')
            self.pareto_metrics_test.plot()
            
            # Plot metrics
            self.logger.log(f"Plotting metrics for round {current_round}")
            utils.plot_metrics(self.eval_round_numbers, roundwise_report, self.output_dir, self.task_name)

        self.logger.log(f"Calculating Pareto metrics using eval scores for round {current_round}")
        self.pareto_metrics_eval.calculate(current_round, self.top_candidates, score_attr='eval_score', token_attr='token_length')
        self.pareto_metrics_eval.plot()

        # Add new candidates to the prompt graph
        self.expansion_graph.update_nodes(self.top_candidates)
        self.expansion_graph.visualize(os.path.join(self.output_dir, "prompt_expansion_graph"))
