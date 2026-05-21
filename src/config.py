from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class EnhancerConfig:
    """
    Configuration for the prompt enhancer.
    """
    type: str = field(
        default="sculpt",
        metadata={"help": "Method for prompt expansion.", "choices": ["sculpt", "ape", "opro", "evoprompt"]}
    )
    # OPRO-specific knobs (ignored by sculpt/ape/evoprompt)
    top_k_history: int = field(default=5, metadata={"help": "OPRO: top-K past prompts (by eval_score) used as fewshot history."})
    n_opro_candidates: int = field(default=4, metadata={"help": "OPRO: number of candidate prompts to generate per parent."})
    opro_temperature: float = field(default=0.7, metadata={"help": "OPRO: sampling temperature for candidate diversity (sculpt uses temperature=0)."})
    # EvoPrompt-specific knobs (ignored by sculpt/ape/opro)
    n_evoprompt_candidates: int = field(default=4, metadata={"help": "EvoPrompt: number of DE offspring to generate per parent."})
    evoprompt_temperature: float = field(default=0.5, metadata={"help": "EvoPrompt: sampling temperature for the DE operator."})
    disable: bool = field(default=False, metadata={"help": "Disable the enhancer."})
    model: str = field(default="gpt4omini", metadata={"help": "Generator model."})
    provider: str = field(default="azure", metadata={"help": "Provider for LLMs.", "choices": ["openai", "azure", "ollama", "codex"]})
    reasoning_effort: str = field(default="low", metadata={"help": "Reasoning effort for codex provider; ignored by others. NOTE: 'minimal' is rejected by gpt-5.x in codex-cli 0.130.0 because the built-in image_gen tool requires effort >= low; use 'low' as the practical floor.", "choices": ["minimal", "low", "medium", "high"]})
    service_tier: str = field(default="fast", metadata={"help": "Service tier for codex provider; ignored by others.", "choices": ["fast", "flex"]})
    max_tokens: int = field(default=4000, metadata={"help": "Maximum tokens for generation"})
    max_tokens_critic: int = field(default=5000, metadata={"help": "Maximum tokens for critic"})
    temperature: float = field(default=0, metadata={"help": "Temperature"})
    top_p: float = field(default=0.01, metadata={"help": "top_p"})
    n_critic: int = field(default=2, metadata={"help": "Number of critics to run."})
    errors_per_critic: int = field(default=4, metadata={"help": "Maximum errors per critic to consider."})
    max_structural_rounds: int = field(default=1, metadata={"help": "Maximum number of structural rounds."})
    # Aggregation
    aggregate_feedbacks: str = field(
        default="no_agg",
        metadata={"help": "Feedback aggregation strategy.", "choices": ["no_agg", "node_agg", "similarity_agg", "diversity_agg"]}
    )
    min_similarity_threshold: float = field(default=0.8, metadata={"help": "Minimum cosine similarity threshold to consider two feedbacks similar"})
    aggregation_factor: float = field(default=0.5, metadata={"help": "Lower the value higher the aggregation"})
    encoder_model: str = field(default="multi-qa-mpnet-base-dot-v1", metadata={"help": "Encoder for measuring semantic similarity.",
                                    "choices": ["multi-qa-mpnet-base-dot-v1", "multi-qa-MiniLM-L6-cos-v1", "all-MiniLM-L6-v2"]})

@dataclass
class OrchestratorConfig:
    """
    Configuration for the orchestrator.
    """
    n_val_exs: int = field(default=1000, metadata={"help": "Number of validation examples."})
    n_test_exs: int = field(default=1000, metadata={"help": "Number of test examples."})
    format_prompt: bool = field(default=False, metadata={"help": "Format prompt to have markdown format"})
    minibatch_size: int = field(default=32, metadata={"help": "Batch size for training evaluations."})
    roe_batch_size: int = field(default=16, metadata={"help": "Batch size for reject-on-error evaluations."})
    n_enchanement: int = field(default=1, metadata={"help": "Number of enhancement iterations."})
    max_expansion_factor: int = field(default=8, metadata={"help": "Maximum expansion factor."})
    sample_type: str = field(
        default="random",
        metadata={"help": "Type of sampling to use.", "choices": ["random", "classwise"]}
    )
    reject_on_errors: bool = field(default=True, metadata={"help": "Reject candidates with errors."})
    n_initial_prompts: int = field(default=1, metadata={"help": "Number of initial prompts."})
    num_fewshots: int = field(default=20, metadata={"help": "Number of few-shot examples used during initial prompt generation."})
    init_prompt_gen: bool = field(default=True, metadata={"help": "Generate initial prompts."})
    max_threads: int = field(default=4, metadata={"help": "Maximum number of threads during expansion or compression."})
    model: str = field(default="gpt4omini", metadata={"help": "Generator model."})
    max_tokens: int = field(default=4000, metadata={"help": "Maximum tokens for generation"})
    temperature: float = field(default=0, metadata={"help": "Temperature"})
    top_p: float = field(default=0.01, metadata={"help": "top_p"})
    provider: str = field(default="azure", metadata={"help": "Provider for LLMs.", "choices": ["openai", "azure", "ollama", "codex"]})
    reasoning_effort: str = field(default="low", metadata={"help": "Reasoning effort for codex provider; ignored by others. NOTE: 'minimal' is rejected by gpt-5.x in codex-cli 0.130.0 because the built-in image_gen tool requires effort >= low; use 'low' as the practical floor.", "choices": ["minimal", "low", "medium", "high"]})
    service_tier: str = field(default="fast", metadata={"help": "Service tier for codex provider; ignored by others.", "choices": ["fast", "flex"]})

@dataclass
class SelectorConfig:
    """
    Configuration for the prompt selector.
    """
    method: str = field(
        default="score",
        metadata={"help": "Candidate selection strategy.", "choices": ["score", "token", "weighted", "nsga2", "nsga2-lcb"]}
    )
    beam_size: int = field(default=4, metadata={"help": "Beam size for candidate selection."})
    weight_selection: float = field(default=1.0, metadata={"help": "Weight for eval_score in weighted selection."})
    score_weight: float = field(default=0.5, metadata={"help": "Weight for eval_score in weighted selection."})
    token_weight: float = field(default=0.5, metadata={"help": "Weight for token_length in weighted selection."})
    prompt_min_length: int = field(default=10, metadata={"help": "Minimum prompt length."})
    prompt_max_length: int = field(default=20000, metadata={"help": "Maximum prompt length."})
    accuracy_buffer: float = field(default=0.0, metadata={"help": "Buffer for accuracy in NSGA-II."})
    token_buffer: float = field(default=0.0, metadata={"help": "Buffer for token length in NSGA-II."})
    k: int = field(default=1, metadata={"help": "K value for scaling the buffer size."})
    filter_factor: float = field(default=1.0, metadata={"help": "Filter factor for filtering bad candidates. The less the value, the more candidates are filtered."})
    lambda_factor: int = field(default=2, metadata={"help": "Lambda factor for scaling the beam size."})
    beta: float = field(default=0.5, metadata={"help": "Beta factor for lower bounding in NSGA-II."})

@dataclass
class EvaluatorConfig:
    """
    Configuration for the prompt evaluator.
    """
    max_threads: int = field(default=4, metadata={"help": "Maximum number of threads during evaluation."})
    max_tokens: int = field(default=4000, metadata={"help": "Maximum tokens for generation"})
    temperature: float = field(default=0, metadata={"help": "Temperature"})
    top_p: float = field(default=0.01, metadata={"help": "top_p"})
    model: str = field(default="gpt4omini", metadata={"help": "Generator model."})
    provider: str = field(default="azure", metadata={"help": "Provider for LLMs.", "choices": ["openai", "azure", "ollama", "codex"]})
    reasoning_effort: str = field(default="low", metadata={"help": "Reasoning effort for codex provider; ignored by others. NOTE: 'minimal' is rejected by gpt-5.x in codex-cli 0.130.0 because the built-in image_gen tool requires effort >= low; use 'low' as the practical floor.", "choices": ["minimal", "low", "medium", "high"]})
    service_tier: str = field(default="fast", metadata={"help": "Service tier for codex provider; ignored by others.", "choices": ["fast", "flex"]})

    validation_type: str = field(default="brute_force", metadata={"help": "Evaluator type.", "choices": ["ucb", "brute_force"]})
    ucb_type: str = field(default="ucb", metadata={"help": "Evaluator type.", "choices": ["ucb", "ucb-e", "ucb-rs", "ucb-mo", "ucb-ws"]})
    samples_per_eval: int = field(default=32, metadata={"help": "Number of samples per evaluation."})
    c: float = field(default=1.0, metadata={"help": "Exploration parameter for UCB."})
    knn_k: int = field(default=2, metadata={"help": "K value for KNN approach."})
    knn_t: float = field(default=0.993, metadata={"help": "Threshold for KNN approach."})
    eval_rounds: int = field(default=8, metadata={"help": "Number of evaluation rounds."})
    eval_prompts_per_round: int = field(default=8, metadata={"help": "Number of prompts per evaluation round."})
    use_stratified_validation: bool = field(default=False, metadata={"help": "Use stratified validation subsets."})
    embedding_model: str = field(default="all-MiniLM-L6-v2", metadata={"help": "Encoder for creating validation subsets."})

@dataclass
class CompressorConfig:
    """
    Configuration for the prompt compressor.
    """
    max_threads: int = field(default=4, metadata={"help": "Maximum number of threads during evaluation."})
    methods: str = field(default="coefficient", metadata={"help": "Compression strategies comma separated."})
    max_tokens: int = field(default=4000, metadata={"help": "Maximum tokens for generation"})
    temperature: float = field(default=0, metadata={"help": "Temperature"})
    top_p: float = field(default=0.01, metadata={"help": "top_p"})
    model: str = field(default="gpt4omini", metadata={"help": "Generator model."})
    max_ratio: float = field(default=0.9, metadata={"help": "Maximum compression ratio."})
    min_ratio: float = field(default=0.5, metadata={"help": "Minimum compression ratio."})
    num_compressions: int = field(default=8, metadata={"help": "Number of compressions."})
    comp_analysis_path: str = field(default="", metadata={"help": "for debugging purposes only."})
    comp_actions: str = field(default="", metadata={"help": "for debugging purposes only."})
    load_structured_prompt_path: str = field(default="", metadata={"help": "for debugging purposes only."})    
    provider: str = field(default="azure", metadata={"help": "Provider for LLMs.", "choices": ["openai", "azure", "ollama", "codex"]})
    reasoning_effort: str = field(default="low", metadata={"help": "Reasoning effort for codex provider; ignored by others. NOTE: 'minimal' is rejected by gpt-5.x in codex-cli 0.130.0 because the built-in image_gen tool requires effort >= low; use 'low' as the practical floor.", "choices": ["minimal", "low", "medium", "high"]})
    service_tier: str = field(default="fast", metadata={"help": "Service tier for codex provider; ignored by others.", "choices": ["fast", "flex"]})

@dataclass
class MutatorConfig:
    """
    Configuration for the prompt mutator.
    """
    max_tokens: int = field(default=4000, metadata={"help": "Maximum tokens for mutation"})
    temperature: float = field(default=0.7, metadata={"help": "Temperature for mutation"})
    top_p: float = field(default=0.9, metadata={"help": "top_p for mutation"})
    model: str = field(default="gpt4omini", metadata={"help": "Mutator model."})
    num_mutations: int = field(default=2, metadata={"help": "Number of mutations to generate."})
    provider: str = field(default="azure", metadata={"help": "Provider for LLMs.", "choices": ["openai", "azure", "ollama", "codex"]})
    reasoning_effort: str = field(default="low", metadata={"help": "Reasoning effort for codex provider; ignored by others. NOTE: 'minimal' is rejected by gpt-5.x in codex-cli 0.130.0 because the built-in image_gen tool requires effort >= low; use 'low' as the practical floor.", "choices": ["minimal", "low", "medium", "high"]})
    service_tier: str = field(default="fast", metadata={"help": "Service tier for codex provider; ignored by others.", "choices": ["fast", "flex"]})

@dataclass
class CacheConfig:
    """
    Configuration for LLM response caching with memory management.
    """
    enabled: bool = field(default=True, metadata={"help": "Enable LLM response caching."})
    cache_dir: str = field(default=".llm_cache", metadata={"help": "Directory to store cache files."})
    refresh_cache: bool = field(default=False, metadata={"help": "If True, ignore existing cache and refresh all responses."})
    clear_on_start: bool = field(default=False, metadata={"help": "Clear cache when starting a new run."})
    print_stats: bool = field(default=True, metadata={"help": "Print cache statistics at the end of run."})
    max_cache_size_mb: int = field(default=51200, metadata={"help": "Maximum cache size in MB before cleanup (default: 5GB)."})
    max_response_size_kb: int = field(default=20480, metadata={"help": "Maximum individual response size in KB to cache (default: 2MB)."})
    max_memory_entries: int = field(default=15000, metadata={"help": "Maximum number of entries to keep in memory LRU cache (default: 5000)."})

@dataclass
class WandbConfig:
    """
    Configuration for Weights & Biases logging.
    """
    enabled: bool = field(default=False, metadata={"help": "Enable wandb logging."})
    project: str = field(default="craft", metadata={"help": "Wandb project name."})
    entity: str = field(default="", metadata={"help": "Wandb entity (team/username); leave empty to use your own default."})
    tags: str = field(default="", metadata={"help": "Comma-separated tags for the run."})
    notes: str = field(default="", metadata={"help": "Notes for the run."})
    log_frequency: int = field(default=1, metadata={"help": "Log metrics every N rounds."})
    log_artifacts: bool = field(default=True, metadata={"help": "Log prompt candidates as artifacts."})
    log_pareto_plots: bool = field(default=True, metadata={"help": "Log Pareto front plots."})
    log_validation_details: bool = field(default=True, metadata={"help": "Log detailed validation metrics."})
    log_test_details: bool = field(default=True, metadata={"help": "Log detailed test metrics."})
    log_candidate_texts: bool = field(default=True, metadata={"help": "Log full candidate prompt texts (may be verbose)."})

@dataclass
class Config:
    """
    Main configuration class.
    """
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    enhancer: EnhancerConfig = field(default_factory=EnhancerConfig)
    compressor: CompressorConfig = field(default_factory=CompressorConfig)
    mutator: MutatorConfig = field(default_factory=MutatorConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_dir: str = field(default=".", metadata={"help": "Path to output directory."})
    task_name: str = field(default="disambiguation_qa", metadata={"help": "Name of the task."})
    run_name: str = field(default="disambiguation_qa", metadata={"help": "Run name for the experiment."})
    max_rounds: int = field(default=8, metadata={"help": "Number of optimization rounds."})
    run_test_evals: bool = field(default=True, metadata={"help": "Run test evaluations."})
    random_seed: int = field(default=42, metadata={"help": "Random seed for reproducibility and cache control."})

# Register configuration with Hydra.
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=Config)