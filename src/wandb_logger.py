import os
import json
import warnings
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Handle wandb import gracefully
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. Logging will be disabled.")


class WandbLogger:
    """
    Centralized wandb logging functionality for ParetoFront project.
    Handles experiment tracking, metrics logging, artifact management, and visualization.
    """
    
    def __init__(self, config, project_root: str, output_dir: str, logger=None):
        """
        Initialize wandb logger with configuration.
        
        Args:
            config: WandbConfig object with wandb settings
            project_root: Root directory of the project
            output_dir: Output directory for the current run
            logger: Existing logger instance
        """
        self.config = config
        self.project_root = project_root
        self.output_dir = output_dir
        self.logger = logger
        self.enabled = config.enabled and WANDB_AVAILABLE
        self.run = None
        
        if not self.enabled:
            self._log("Wandb logging disabled or unavailable")
            return
            
        # Parse tags
        self.tags = [tag.strip() for tag in config.tags.split(',') if tag.strip()] if config.tags else []
        
    def _log(self, message: str):
        """Helper to log messages using existing logger or print."""
        if self.logger:
            self.logger.log(f"[WANDB] {message}")
        else:
            print(f"[WANDB] {message}")
    
    def init_run(self, task_name: str, run_name: str, run_config: Dict[str, Any]):
        """
        Initialize wandb run with configuration.
        
        Args:
            task_name: Name of the task being optimized
            run_name: Name for this specific run
            run_config: Configuration dictionary to log
        """
        if not self.enabled:
            return
            
        try:
            # Create run name with task info
            full_run_name = f"{task_name}_{run_name}" if run_name != task_name else run_name
            
            # Initialize wandb run.
            # console="off" disables wandb's stdout/stderr capture hook. That
            # hook recursively re-enters on very large prints (e.g. OPRO's
            # full generated prompts), raising "maximum recursion depth
            # exceeded" in wandb/sdk/lib/console_capture.py and crashing the
            # affected candidate's enhancement. We don't need console mirroring
            # in wandb — the run log.txt already captures everything.
            self.run = wandb.init(
                project=self.config.project,
                entity=self.config.entity if self.config.entity else None,
                name=full_run_name,
                tags=self.tags + [task_name],
                notes=self.config.notes,
                config=run_config,
                reinit=True,
                settings=wandb.Settings(console="off"),
            )
            
            self._log(f"Initialized wandb run: {full_run_name}")
            
        except Exception as e:
            self._log(f"Failed to initialize wandb: {e}")
            self.enabled = False
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None, prefix: str = ""):
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number (typically round number)
            prefix: Prefix to add to metric names
        """
        if not self.enabled or not metrics:
            return
            
        try:
            # Add prefix to metric names if provided
            if prefix:
                metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            
            wandb.log(metrics, step=step)
            
        except Exception as e:
            self._log(f"Failed to log metrics: {e}")
    
    def log_validation_metrics(self, current_round: int, candidates: List[Any], validation_report_path: str):
        """
        Log validation metrics and candidate information.
        
        Args:
            current_round: Current optimization round
            candidates: List of candidate objects with eval_score and token_length
            validation_report_path: Path to validation report file
        """
        if not self.enabled:
            return
            
        try:
            # Extract candidate metrics
            eval_scores = [getattr(c, 'eval_score', 0) for c in candidates]
            eval_stds = [getattr(c, 'eval_std', 0) for c in candidates]
            token_lengths = [getattr(c, 'token_length', 0) for c in candidates]
            candidate_ids = [getattr(c, 'id', i) for i, c in enumerate(candidates)]
            
            # Calculate validation metrics
            validation_metrics = {
                'validation/num_candidates': len(candidates),
                'validation/mean_eval_score': np.mean(eval_scores) if eval_scores else 0,
                'validation/max_eval_score': np.max(eval_scores) if eval_scores else 0,
                'validation/min_eval_score': np.min(eval_scores) if eval_scores else 0,
                'validation/std_eval_score': np.std(eval_scores) if len(eval_scores) > 1 else 0,
                'validation/mean_eval_std': np.mean(eval_stds) if eval_stds else 0,
                'validation/max_eval_std': np.max(eval_stds) if eval_stds else 0,
                'validation/min_eval_std': np.min(eval_stds) if eval_stds else 0,
                'validation/mean_token_length': np.mean(token_lengths) if token_lengths else 0,
                'validation/min_token_length': np.min(token_lengths) if token_lengths else 0,
                'validation/max_token_length': np.max(token_lengths) if token_lengths else 0,
                'validation/std_token_length': np.std(token_lengths) if len(token_lengths) > 1 else 0,
            }
            
            self.log_metrics(validation_metrics, step=current_round)
            
            # Log detailed candidate information if enabled
            if self.config.log_validation_details:
                candidate_data = []
                for i, candidate in enumerate(candidates):
                    candidate_data.append({
                        'candidate_id': candidate_ids[i],
                        'eval_score': eval_scores[i],
                        'eval_std': eval_stds[i],
                        'token_length': token_lengths[i],
                        'round': current_round,
                        'parent_id': getattr(candidate, 'parent_id', None)
                    })
                
                # Create wandb table for candidate details
                table = wandb.Table(
                    columns=['candidate_id', 'eval_score', 'eval_std', 'token_length', 'round', 'parent_id'],
                    data=[[row['candidate_id'], row['eval_score'], row['eval_std'], row['token_length'], 
                           row['round'], row['parent_id']] for row in candidate_data]
                )
                
                wandb.log({f'validation/candidates_round_{current_round}': table}, step=current_round)
            
            # Log validation report as artifact if file exists
            if self.config.log_artifacts and os.path.exists(validation_report_path):
                artifact = wandb.Artifact(
                    name=f'validation_report_round_{current_round}',
                    type='validation_report'
                )
                artifact.add_file(validation_report_path)
                wandb.log_artifact(artifact)
            
        except Exception as e:
            self._log(f"Failed to log validation metrics: {e}")
    
    def log_test_metrics(self, current_round: int, candidates: List[Any], test_reports: List[Dict], test_report_path: str):
        """
        Log test evaluation metrics and detailed results.
        
        Args:
            current_round: Current optimization round
            candidates: List of candidate objects with test_score
            test_reports: List of test evaluation reports
            test_report_path: Path to merged test report file
        """
        if not self.enabled:
            return
            
        try:
            # Extract test metrics
            test_scores = [getattr(c, 'test_score', 0) for c in candidates]
            candidate_ids = [getattr(c, 'id', i) for i, c in enumerate(candidates)]
            token_lengths = [getattr(c, 'token_length', 0) for c in candidates]
            
            # Calculate test metrics
            test_metrics = {
                'test/num_candidates': len(candidates),
                'test/mean_test_score': np.mean(test_scores) if test_scores else 0,
                'test/max_test_score': np.max(test_scores) if test_scores else 0,
                'test/min_test_score': np.min(test_scores) if test_scores else 0,
                'test/std_test_score': np.std(test_scores) if len(test_scores) > 1 else 0,
            }
            
            self.log_metrics(test_metrics, step=current_round)
            
            # Log detailed test results if enabled
            if self.config.log_test_details:
                test_data = []
                for i, candidate in enumerate(candidates):
                    test_data.append({
                        'candidate_id': candidate_ids[i],
                        'test_score': test_scores[i],
                        'token_length': token_lengths[i],
                        'round': current_round
                    })
                
                # Create wandb table for test results
                table = wandb.Table(
                    columns=['candidate_id', 'test_score', 'token_length', 'round'],
                    data=[[row['candidate_id'], row['test_score'], row['token_length'], 
                           row['round']] for row in test_data]
                )
                
                wandb.log({f'test/results_round_{current_round}': table}, step=current_round)
            
            # Log test report as artifact
            if self.config.log_artifacts and os.path.exists(test_report_path):
                artifact = wandb.Artifact(
                    name=f'test_report_round_{current_round}',
                    type='test_report'
                )
                artifact.add_file(test_report_path)
                wandb.log_artifact(artifact)
            
        except Exception as e:
            self._log(f"Failed to log test metrics: {e}")
    
    def log_selected_candidates(self, current_round: int, selected_candidates: List[Any]):
        """
        Log information about selected candidates.
        
        Args:
            current_round: Current optimization round
            selected_candidates: List of selected candidate objects
        """
        if not self.enabled:
            return
            
        try:
            # Extract candidate information
            candidate_info = []
            for i, candidate in enumerate(selected_candidates):
                info = {
                    'candidate_id': getattr(candidate, 'id', i),
                    'eval_score': getattr(candidate, 'eval_score', 0),
                    'eval_std': getattr(candidate, 'eval_std', 0),
                    'token_length': getattr(candidate, 'token_length', 0),
                    'rank': i + 1,
                    'round': current_round,
                    'parent_id': getattr(candidate, 'parent_id', None)
                }
                candidate_info.append(info)
            
            # Log selection metrics
            selection_metrics = {
                'selection/num_selected': len(selected_candidates),
                'selection/best_eval_score': candidate_info[0]['eval_score'] if candidate_info else 0,
                'selection/worst_eval_score': candidate_info[-1]['eval_score'] if candidate_info else 0,
                'selection/best_token_length': min(info['token_length'] for info in candidate_info) if candidate_info else 0,
                'selection/worst_token_length': max(info['token_length'] for info in candidate_info) if candidate_info else 0,
            }
            
            self.log_metrics(selection_metrics, step=current_round)
            
            # Log detailed selection table
            table = wandb.Table(
                columns=['candidate_id', 'eval_score', 'eval_std', 'token_length', 'rank', 'round', 'parent_id'],
                data=[[info['candidate_id'], info['eval_score'], info['eval_std'], info['token_length'], 
                       info['rank'], info['round'], info['parent_id']] for info in candidate_info]
            )
            
            wandb.log({f'selection/candidates_round_{current_round}': table}, step=current_round)
            
            # Log candidate texts if enabled
            if self.config.log_candidate_texts:
                for i, candidate in enumerate(selected_candidates):
                    try:
                        prompt_text = candidate.get_prompt_chat_template()
                        wandb.log({
                            f'selection/candidate_{candidate.id}_text': wandb.Html(f"<pre>{prompt_text}</pre>")
                        }, step=current_round)
                    except Exception as e:
                        self._log(f"Failed to log candidate text for {candidate.id}: {e}")
            
        except Exception as e:
            self._log(f"Failed to log selected candidates: {e}")
    
    def log_pareto_metrics(self, current_round: int, pareto_metrics: Dict[str, Any], 
                          candidates: List[Any], score_attr: str = 'test_score'):
        """
        Log Pareto optimization metrics and visualizations.
        
        Args:
            current_round: Current optimization round
            pareto_metrics: Dictionary of calculated Pareto metrics
            candidates: List of candidate objects
            score_attr: Attribute name for scores (test_score or eval_score)
        """
        if not self.enabled:
            return
            
        try:
            # Log core Pareto metrics
            metrics_to_log = {}
            for key, value in pareto_metrics.items():
                if key != 'round':
                    metrics_to_log[f'pareto/{key}'] = value
            
            self.log_metrics(metrics_to_log, step=current_round)
            
            # Create and log Pareto front visualization if enabled
            if self.config.log_pareto_plots and candidates:
                self._create_and_log_pareto_plot(current_round, candidates, score_attr)
            
        except Exception as e:
            self._log(f"Failed to log Pareto metrics: {e}")
    
    def _create_and_log_pareto_plot(self, current_round: int, candidates: List[Any], score_attr: str):
        """Create and log Pareto front visualization."""
        try:
            scores = [getattr(c, score_attr, 0) for c in candidates]
            tokens = [getattr(c, 'token_length', 0) for c in candidates]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(tokens, scores, alpha=0.7, s=50)
            plt.xlabel('Token Count')
            plt.ylabel('Score')
            plt.title(f'Pareto Front - Round {current_round}')
            plt.grid(True, alpha=0.3)
            
            # Invert x-axis since we want fewer tokens
            plt.gca().invert_xaxis()
            
            # Log the plot
            wandb.log({f'pareto/front_round_{current_round}': wandb.Image(plt)}, step=current_round)
            plt.close()
            
        except Exception as e:
            self._log(f"Failed to create Pareto plot: {e}")
    
    def log_round_summary(self, current_round: int, round_info: Dict[str, Any]):
        """
        Log summary information for a complete round.
        
        Args:
            current_round: Current optimization round
            round_info: Dictionary with round summary information
        """
        if not self.enabled:
            return
            
        try:
            summary_metrics = {}
            for key, value in round_info.items():
                summary_metrics[f'round/{key}'] = value
            
            self.log_metrics(summary_metrics, step=current_round)
            
        except Exception as e:
            self._log(f"Failed to log round summary: {e}")
    
    def log_artifacts(self, current_round: int, round_dir: str):
        """
        Log important files as wandb artifacts.
        
        Args:
            current_round: Current optimization round
            round_dir: Directory containing round outputs
        """
        if not self.enabled or not self.config.log_artifacts:
            return
            
        try:
            # Create round artifact
            artifact = wandb.Artifact(
                name=f'round_{current_round}_outputs',
                type='round_outputs',
                description=f'Outputs from optimization round {current_round}'
            )
            
            # Add candidate files
            for filename in os.listdir(round_dir):
                if filename.endswith('.txt') or filename.endswith('.tsv') or filename.endswith('.json'):
                    file_path = os.path.join(round_dir, filename)
                    if os.path.isfile(file_path):
                        artifact.add_file(file_path, name=filename)
            
            wandb.log_artifact(artifact)
            
        except Exception as e:
            self._log(f"Failed to log artifacts: {e}")
    
    def finish_run(self):
        """Finish the wandb run."""
        if not self.enabled or not self.run:
            return
            
        try:
            wandb.finish()
            self._log("Wandb run finished")
            
        except Exception as e:
            self._log(f"Failed to finish wandb run: {e}")
    
    def log_config_update(self, config_dict: Dict[str, Any]):
        """Update wandb config with additional information."""
        if not self.enabled or not self.run:
            return
            
        try:
            wandb.config.update(config_dict)
            
        except Exception as e:
            self._log(f"Failed to update wandb config: {e}")