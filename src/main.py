import os
import sys
import time
import json
import importlib
from collections import Counter
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import yaml
import uuid
import random
import numpy as np
import utils
from config import Config
from orchestrator import Orchestrator
from wandb_logger import WandbLogger
from llm_cache import initialize_cache, print_global_cache_stats
import hashlib
from omegaconf import OmegaConf

# Start tracemalloc early so the per-round memory diagnostic can take
# meaningful snapshots. Opt-in via PARETO_MEM_DIAG=1 to avoid overhead
# in production runs.
if os.environ.get("PARETO_MEM_DIAG", "0") == "1":
    import tracemalloc as _tm
    _tm.start(25)
    print("[MEM] tracemalloc started (frame depth 25)")

_SELECTOR_UCB_PAIRING = {
    "score": "ucb-e",
    "token": "ucb-e",
    "weighted": "ucb-ws",
    "nsga2": "ucb-mo",
    "nsga2-lcb": "ucb-mo",
}


def _validate_selector_ucb_pairing(config):
    """Fail fast when selector.method and evaluator.ucb_type are paired incorrectly.

    Past incident: codex baselines were launched with selector=score + ucb-mo,
    which silently runs Pareto-aware validation on a single-objective beam.
    """
    if config.evaluator.validation_type != "ucb":
        return
    expected = _SELECTOR_UCB_PAIRING.get(config.selector.method)
    if expected is None or config.evaluator.ucb_type == expected:
        return
    raise ValueError(
        f"Config mismatch: selector.method={config.selector.method!r} expects "
        f"evaluator.ucb_type={expected!r} but got {config.evaluator.ucb_type!r}. "
        f"Pairings: {_SELECTOR_UCB_PAIRING}"
    )


def setup_random_seed(seed):
    """
    Set up random seed for reproducibility across all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Also set seed in all modules that have their own random.seed() calls
    # This ensures consistency across the entire codebase
    print(f"Random seed set to: {seed}")



def setup_cache(config, logger):
    """
    Initialize the LLM cache with configuration settings and memory management.

    The cache task_name is now a deterministic fingerprint (SHA1) of the
    significant parts of the config (with a few volatile fields removed).
    Any change to any included parameter will change the fingerprint and
    therefore produce a new cache (as requested).
    """
    # Use cache_dir directly without nesting it in run-specific output_dir
    cache_dir = config.cache.cache_dir

    # Build a deterministic fingerprint of the config so any single parameter
    # change yields a new cache task name.
    # Convert OmegaConf to a plain dict, resolve interpolations.
    cfg_dict = OmegaConf.to_container(config, resolve=True)

    # Remove volatile or non-config fields that shouldn't change cache:
    # e.g., output paths, run-specific metadata, and large logs.
    for volatile_key in ("output_dir", "run_name", "metadata", "project_root"):
        cfg_dict.pop(volatile_key, None)

    # Also remove wandb credentials / runtime-only sections that shouldn't affect the cache.
    # (Keep wandb *settings* if you want them to be part of the fingerprint; adjust as desired.)
    cfg_dict.pop("wandb", None)

    # Serialize deterministically and hash.
    cfg_json = json.dumps(cfg_dict, sort_keys=True, default=str)
    fingerprint = hashlib.sha1(cfg_json.encode("utf-8")).hexdigest()

    # Compose a human-readable task_name that includes the task name, seed, and fingerprint.
    # Using a short slice of the fingerprint keeps filenames manageable.
    cache_task_name = f"{config.task_name}_seed{config.random_seed}_{fingerprint[:12]}"

    logger.log(f"Initializing LLM cache: dir={cache_dir}, task_name={cache_task_name}, fingerprint={fingerprint}")

    # Initialize cache with configuration, task name, and memory management settings
    cache = initialize_cache(
        cache_dir=cache_dir,
        enabled=config.cache.enabled,
        random_seed=config.random_seed,
        refresh_cache=config.cache.refresh_cache,
        task_name=cache_task_name,
        max_cache_size_mb=config.cache.max_cache_size_mb,
        max_response_size_kb=config.cache.max_response_size_kb,
        max_memory_entries=getattr(config.cache, "max_memory_entries", 5000),
        logger=logger
    )

    # Optionally clear on start
    if config.cache.clear_on_start:
        logger.log("Clearing existing cache as requested...")
        cache.clear_cache()
        logger.log("Cache cleared.")

    return cache

def load_task_and_evaluator(task_name, config, project_root, logger):
    # Load Task module dynamically
    task_module_name = f"scenario.{task_name}.task"
    try:
        task_module = importlib.import_module(task_module_name)
    except ImportError as e:
        logger.log(f"Error importing module '{task_module_name}': {e}")
        sys.exit(1)

    task_class = getattr(task_module, 'Task', None)
    if task_class is None:
        logger.log(f"No 'Task' class found in module '{task_module_name}'")
        sys.exit(1)

    data_dir = os.path.join(project_root, 'scenario', task_name, 'data')
    # Try to initialize task with random seed if supported
    try:
        task_instance = task_class(data_dir, random_seed=config.random_seed)
    except TypeError:
        # Fallback to original constructor if random_seed parameter is not supported
        task_instance = task_class(data_dir)
        # Then try to reseed if method is available
        if hasattr(task_instance, 'reseed'):
            task_instance.reseed(config.random_seed)

    task_config = None
    default_config_path = os.path.join(project_root, 'scenario', task_name, 'config.yaml')
    if os.path.exists(default_config_path):
        with open(default_config_path, 'r', encoding='utf-8') as file:
            task_config = yaml.safe_load(file)

    # Load Evaluator module dynamically
    eval_manager_name = f"scenario.{task_name}.evaluation_manager"
    try:
        eval_manager_module = importlib.import_module(eval_manager_name)
    except ImportError as e:
        logger.log(f"No eval.py module found; evaluation won't run with error {e}.")
        return task_instance, None, None

    eval_manager_class = getattr(eval_manager_module, 'EvaluationManager', None)
    if eval_manager_class is None:
        logger.log("No 'EvaluationManager' class found; evaluation won't run.")
        return task_instance, None, None

    base_eval_manager = eval_manager_class(task_name, config.evaluator, logger, task_config) if task_config else None
    return task_instance, base_eval_manager, task_config

def main():
    default_cfg = OmegaConf.structured(Config)
    cli_cfg = OmegaConf.from_cli()
    config = OmegaConf.merge(default_cfg, cli_cfg)

    _validate_selector_ucb_pairing(config)

    # Set up random seed for reproducibility
    setup_random_seed(config.random_seed)

    # Create output directory with timestamp.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    metadata = f"{uuid.uuid4().hex[:8]}"
    output_dir = os.path.join(config.output_dir, "output", config.task_name, f"{config.run_name}-{metadata}")
    os.makedirs(output_dir, exist_ok=True)

    # Update cache directory to be within output directory
    config.output_dir = output_dir

    OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))

    # Initialize log file.
    log_filepath = os.path.join(output_dir, "log.txt")
    if os.path.exists(log_filepath):
        os.remove(log_filepath)

    # Initialize logger.
    logger = utils.Logger(log_filepath)
    logger.log("Loading configuration:")
    logger.log(OmegaConf.to_yaml(config))

    # Set up LLM cache
    cache = setup_cache(config, logger)

    # Initialize wandb logging
    wandb_logger = WandbLogger(config.wandb, project_root, output_dir, logger)
    
    # Prepare config for wandb
    wandb_config = OmegaConf.to_container(config, resolve=True)
    wandb_config['output_dir'] = output_dir
    wandb_config['metadata'] = metadata
    wandb_config['project_root'] = project_root
    
    wandb_logger.init_run(config.task_name, config.run_name, wandb_config)

    component_keys = ['expansion', 'evaluation', 'scorer']
    overall_token_usage = utils.TokenUsageDetails()
    overall_token_usage.init_components(component_keys)
    overall_action_counter = Counter()
    current_round = 0

    # Initialize task and eval_manager
    task, eval_manager, task_config = load_task_and_evaluator(config.task_name, config, project_root, logger)

    with open(os.path.join(output_dir, "task_config.yaml"), 'w', encoding='utf-8') as file:
        yaml.safe_dump(task_config, file, default_flow_style=False)

    logger.log("Reading task configuration:")
    logger.log(json.dumps(task_config, indent=4))

    # Initialize orchestrator
    orchestrator = Orchestrator(config.task_name, task, eval_manager, config, project_root, output_dir, logger, wandb_logger, random_seed=config.random_seed)

    # Load human-written initial prompts
    orchestrator.read_initial_prompts(task_config["initial_prompts"])

    # Cold Start by generating initial prompts
    if orchestrator.config.init_prompt_gen:
        current_round_dir = os.path.join(output_dir, "cold_start_round")
        os.mkdir(current_round_dir)
        orchestrator.generate_initial_prompts(current_round_dir)
        orchestrator.process_candidates(current_round_dir, 0, config.run_test_evals)
        current_round += 1
        config.max_rounds += 1  # To account for the cold start round

    # Structure human-written initial prompts
    orchestrator.structure_initial_prompts(current_round)
    orchestration_mode = False
    while current_round < config.max_rounds + 1:
        logger.log(f"___STARTING ROUND {current_round}___")
        
        # Log round start to wandb
        if wandb_logger:
            round_info = {
                'round_start': current_round,
                'num_top_candidates': len(orchestrator.top_candidates) if hasattr(orchestrator, 'top_candidates') else 0
            }
            wandb_logger.log_round_summary(current_round, round_info)

        # Create round directory.
        current_round_dir = os.path.join(output_dir, f"round_{current_round}")
        os.mkdir(current_round_dir)

        # Initialize token usage tracking for the round.
        round_token_usage = utils.TokenUsageDetails()
        round_token_usage.init_components(['overall'])
        round_token_usage.add_token_usage_for_component(utils.TokenUsage(), 'overall')

        if orchestration_mode:
            sampled_train_batch = orchestrator.sample_train_batch(orchestrator.config.minibatch_size)
            logger.log(f"Sampled {len(sampled_train_batch)} training examples for round {current_round}")
            
            # Log training batch info to wandb
            if wandb_logger:
                batch_info = {
                    'train_batch_size': len(sampled_train_batch),
                    'minibatch_size': orchestrator.config.minibatch_size
                }
                wandb_logger.log_metrics(batch_info, step=current_round, prefix='training')
            
            # Copy the top candidates 
            existing_candidates = orchestrator.top_candidates.copy()
            enhanced_candidates = []
            if orchestrator.enhancer.config.disable is False:
                logger.log(f"Enhancing candidates for round {current_round}")
                enhanced_candidates = orchestrator.enhance_candidates(sampled_train_batch, current_round, current_round_dir)
                logger.log(f"Enhancement complete with {len(enhanced_candidates)} candidates")
                logger.log(f"Enhanced candidate IDs: {[c.id for c in enhanced_candidates]}")

                # Log enhancement results to wandb
                if wandb_logger:
                    enhancement_info = {
                        'num_enhanced_candidates': len(enhanced_candidates),
                        'num_existing_candidates': len(existing_candidates)
                    }
                    wandb_logger.log_metrics(enhancement_info, step=current_round, prefix='enhancement')
            
            compressed_candidates = []
            if orchestrator.prompt_compressor.config.methods != "NA" and orchestrator.prompt_compressor.config.num_compressions > 0:
                logger.log(f"Compressing candidates for round {current_round}")
                compressed_candidates = orchestrator.compress_candidates(sampled_train_batch, current_round, current_round_dir)
                logger.log(f"Compression complete with {len(compressed_candidates)} candidates")
                logger.log(f"Compressed candidate IDs: {[c.id for c in compressed_candidates]}")

                # Log compression results to wandb
                if wandb_logger:
                    compression_info = {
                        'num_compressed_candidates': len(compressed_candidates),
                        'compression_methods': orchestrator.prompt_compressor.config.methods
                    }
                    wandb_logger.log_metrics(compression_info, step=current_round, prefix='compression')

            mutated_candidates = []
            if orchestrator.mutator.config.num_mutations > 0:
                logger.log(f"Mutating candidates for round {current_round}")
                mutated_candidates = orchestrator.mutate_candidates(current_round, current_round_dir)
                logger.log(f"Mutation complete with {len(mutated_candidates)} candidates")
                logger.log(f"Mutated candidate IDs: {[c.id for c in mutated_candidates]}")

                # Log mutation results to wandb
                if wandb_logger:
                    mutation_info = {
                        'num_mutated_candidates': len(mutated_candidates),
                        'num_mutations_per_candidate': orchestrator.mutator.config.num_mutations
                    }
                    wandb_logger.log_metrics(mutation_info, step=current_round, prefix='mutation')
                
            # Deduplicate against previous and update
            new_candidates = enhanced_candidates + mutated_candidates + compressed_candidates
            orchestrator.top_candidates = orchestrator.deduplicate_candidates(existing_candidates, new_candidates)
            logger.log(f"{len(orchestrator.top_candidates)} candidates after dedupe")
            logger.log(f"Top candidate IDs: {[c.id for c in orchestrator.top_candidates]}")
            
            # Log deduplication results to wandb
            if wandb_logger:
                dedup_info = {
                    'total_new_candidates': len(new_candidates),
                    'candidates_after_dedup': len(orchestrator.top_candidates),
                    'dedup_efficiency': len(orchestrator.top_candidates) / len(new_candidates) if new_candidates else 0
                }
                wandb_logger.log_metrics(dedup_info, step=current_round, prefix='deduplication')

        # Process candidates for validation, selection, and evaluation
        orchestrator.process_candidates(current_round_dir, current_round, (current_round == 0 or current_round == config.max_rounds or config.run_test_evals))

        # Log round completion to wandb
        if wandb_logger:
            round_completion_info = {
                'round_completed': current_round,
                'final_candidates_count': len(orchestrator.top_candidates)
            }
            wandb_logger.log_round_summary(current_round, round_completion_info)

            # Log artifacts for this round
            wandb_logger.log_artifacts(current_round, current_round_dir)

        # Per-round memory diagnostic (opt-in via PARETO_MEM_DIAG=1).
        # When enabled, logs deep-size of expansion_graph / top_candidates /
        # error_evaluations / LLMCache buffer + a tracemalloc top-N delta.
        if os.environ.get("PARETO_MEM_DIAG", "0") == "1":
            try:
                from memory_diagnostic import log_round_memory
                log_round_memory(orchestrator, current_round, logger)
            except Exception as _mem_err:
                logger.log(f"[MEM] diagnostic failed: {_mem_err}")

        current_round += 1
        orchestration_mode = True

    logger.log("Experiment complete. Check output directory for results.")
    
    # Print cache statistics if enabled
    if config.cache.print_stats:
        print_global_cache_stats()
        
    # Ensure all cache writes are flushed
    if cache:
        cache.stop()
    
    # Finish wandb run
    wandb_logger.finish_run()

if __name__ == "__main__":
    main()