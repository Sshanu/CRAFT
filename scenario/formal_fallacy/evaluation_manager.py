import re
import json
import numpy as np
from sklearn.metrics import f1_score, classification_report
from src.base_evaluation_manager import BaseEvaluationManager

class EvaluationManager(BaseEvaluationManager):
    def __init__(self, task_name, config, logger, task_config=None):
        """
        Initialize the TaskEvaluator.
        """
        super().__init__(task_name, config, logger, task_config)

    def populate_user_prompt_with_input(self, ex, user_prompt):
        user_prompt = user_prompt.replace("{{input}}", str(ex.get('input', '')))
        return user_prompt
    
    def get_batch_score(self, labels, preds):
        scores = f1_score(labels, preds, average="macro", zero_division=0)
        return scores