import re
import json
import numpy as np
from sklearn.metrics import f1_score, classification_report
from src.base_evaluation_manager import BaseEvaluationManager

def multilabel_to_binary_matrix(y_raw, all_labels):
    binary_matrix = np.zeros((len(y_raw), len(all_labels)), dtype=int)
    for i, labels_list in enumerate(y_raw):
        if not labels_list:  # Handle empty label lists
            # Default to "Safe" if available
            if "Safe" in all_labels:
                binary_matrix[i, all_labels.index("Safe")] = 1
            continue
            
        found_any_label = False
        for label in labels_list:
            try:
                binary_matrix[i, all_labels.index(label)] = 1
                found_any_label = True
            except ValueError:
                print(f"Warning: Unknown label '{label}' in {labels_list}. Skipping.")
                continue
        
        # If no valid labels were found, default to "Safe" if available
        if not found_any_label and "Safe" in all_labels:
            binary_matrix[i, all_labels.index("Safe")] = 1
            
    return binary_matrix

class EvaluationManager(BaseEvaluationManager):
    def __init__(self, task_name, config, logger, task_config=None):
        """
        Initialize the TaskEvaluator.
        """
        super().__init__(task_name, config, logger, task_config)

    def get_true_labels(self, text):
        try:
            # Handle None, empty, or non-string inputs
            if not text or text == "nan" or text == "None":
                return ["Safe"]
                
            text = str(text).strip()
            if not text:
                return ["Safe"]
            
            # Extract JSON content within curly braces
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                return ["Safe"]
            
            json_string = match.group(0)
            data = json.loads(json_string.strip())
            if not isinstance(data, dict):
                return ["Safe"]
                
            # Extract keys where value is True
            harm_list = [key for key, value in data.items() if value]
            return harm_list if harm_list else ["Safe"]
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            self.logger.log(f"Warning: Error parsing label '{text}': {e}. Defaulting to Safe.")
            return ["Safe"]
        except Exception as e:
            self.logger.log(f"Unexpected error parsing label '{text}': {e}. Defaulting to Safe.")
            return ["Safe"]
            
    def get_batch_score(self, labels, preds):
        y_true_raw = [self.get_true_labels(str(label)) for label in labels]
        y_pred_raw = [self.get_true_labels(str(pred)) for pred in preds]
        y_true = multilabel_to_binary_matrix(y_true_raw, self.unique_labels)
        y_pred = multilabel_to_binary_matrix(y_pred_raw, self.unique_labels)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return round(weighted_f1, 2)

    def get_instance_score(self, label, pred):
        y_true_raw = [self.get_true_labels(str(l)) for l in [label]]
        y_pred_raw = [self.get_true_labels(str(p)) for p in [pred]]
        y_true = multilabel_to_binary_matrix(y_true_raw, self.unique_labels)
        y_pred = multilabel_to_binary_matrix(y_pred_raw, self.unique_labels)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return round(weighted_f1, 2)

    def is_misclassified(self, input_text, true_label, predicted_label):
        y_true_raw = self.get_true_labels(str(true_label))
        y_pred_raw = self.get_true_labels(str(predicted_label))
        return set(y_true_raw) != set(y_pred_raw)   
        
    def parse_response(self, response):
        # Ensure response is a string to avoid 'bool' object has no attribute 'lower' error
        response_str = str(response) if response is not None else ""
        if "could not get a response" in response_str.lower():
            return "NA", ""

        answer_json = self.extract_answer(response)
        
        # Initialize defaults
        answer = response
        explanation = ""
        
        try:
            # Handle list response
            if isinstance(answer_json, list) and len(answer_json) > 0:
                answer_json = answer_json[0]
            
            # Handle string response
            if isinstance(answer_json, str):
                answer = answer_json
                explanation = ""
            # Handle dictionary response
            elif isinstance(answer_json, dict):
                answer = answer_json.get("answer", response)
                explanation = answer_json.get("explanation", "")
            else:
                # Fallback for unexpected types
                answer = str(answer_json) if answer_json is not None else response
                explanation = ""
                
        except Exception as e:
            self.logger.log(f"Error parsing response: {e}")
            self.logger.log(f"Answer JSON: {answer_json}, Type: {type(answer_json)}")
            answer = response
            explanation = ""
        
        # self.logger.log(f"Full Response: {response}, Parsed Answer: {answer}, Parsed Explanation: {explanation}")
        return answer, explanation

    def create_report(self, labels, preds):
        y_true_raw = [self.get_true_labels(str(label)) for label in labels]
        y_pred_raw = [self.get_true_labels(str(pred)) for pred in preds]
        y_true = multilabel_to_binary_matrix(y_true_raw, self.unique_labels)
        y_pred = multilabel_to_binary_matrix(y_pred_raw, self.unique_labels)
        
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        try:
            exact_match_accuracy = sum(set(y_t) == set(y_p) for y_t, y_p in zip(y_true_raw, y_pred_raw)) / len(y_true_raw)
        except Exception as e:
            self.logger.log(f"Exception: {e}, True Labels: {y_true_raw}, Predicted Labels: {y_pred_raw}")
            exact_match_accuracy = 0
            self.logger.log("Exact match accuracy set to 0 due to exception")

        # Use processed labels for classification report
        try:
            # Convert binary matrices back to label strings for classification_report
            y_true_labels = []
            y_pred_labels = []
            
            for i in range(len(y_true)):
                true_label_indices = np.where(y_true[i] == 1)[0]
                pred_label_indices = np.where(y_pred[i] == 1)[0]
                
                # Convert indices back to label names
                true_labels = [self.unique_labels[idx] for idx in true_label_indices] if len(true_label_indices) > 0 else ["Safe"]
                pred_labels = [self.unique_labels[idx] for idx in pred_label_indices] if len(pred_label_indices) > 0 else ["Safe"]
                
                # For classification_report, use the primary label (first one or "Safe")
                y_true_labels.append(str(true_labels[0]))
                y_pred_labels.append(str(pred_labels[0]))
            
            report = classification_report(y_true_labels, y_pred_labels, output_dict=True, zero_division=0)
            output_report = {}
            for key, metrics in report.items():
                if key in ["macro avg", "weighted avg", "accuracy"]:
                    continue
                output_report[key] = {
                    "precision": round(metrics["precision"], 2),
                    "recall": round(metrics["recall"], 2),
                    "f1score": round(metrics["f1-score"], 2),
                    "support": metrics["support"]  # support is typically an integer
                }
        except Exception as e:
            self.logger.log(f"Error generating classification report: {e}")
            output_report = {}
            
        output_report["macro_f1"] = round(macro_f1, 2)
        output_report["weighted_f1"] = round(weighted_f1, 2)
        output_report["exact_match_accuracy"] = round(exact_match_accuracy, 2)
        return output_report
            
    def populate_user_prompt_with_input(self, ex, user_prompt):
        user_prompt = user_prompt.replace("{{input}}", str(ex.get('input', '')))
        return user_prompt