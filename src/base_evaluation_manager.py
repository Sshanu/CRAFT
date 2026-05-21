from tqdm import tqdm
import pandas as pd
import requests
import dirtyjson
import llm
import dirtyjson
import llm
import concurrent.futures
from sklearn.metrics import classification_report, accuracy_score
import utils
import os

# Sentinel prediction value for examples that failed at the LLM API layer
# (codex CLI timeouts, safety-filter refusals, JSON-parse failures of empty
# responses). These should NOT count against the candidate's accuracy — they
# indicate infra noise, not a wrong classification. The run_evaluate loop
# filters predictions equal to this sentinel out of the score calculation
# while still recording them in output_exs for inspection.
API_ERROR_LABEL = "__API_ERROR__"


class BaseEvaluationManager:
    """
    Base class for EvaluationManager.
    """
    def __init__(self, task_name, config, logger, task_config=None):
        """
        Initializes the EvaluationManager with task name, config, and task config.

        Args:
            task_name (str): The name of the task.
            config (dict): The configuration dictionary.
            task_config (dict, optional): The task-specific configuration. Defaults to None.
        """
        self.task_name = task_name
        self.config = config
        self.task_config = task_config
        self.logger = logger
        self.aggregation_keys = task_config.get("aggregation_keys", []) if task_config else []
        # Ensure unique_labels are always strings to prevent boolean conversion issues
        raw_labels = task_config.get("unique_labels", []) if task_config else []
        self.unique_labels = [str(label) for label in raw_labels]
        self.global_df = None
        self.ae_template = "src/data/answer_extraction.md"
        
        # Log initialization details
        self.logger.log(f"[INIT] Initializing BaseEvaluationManager for task: {task_name}")
        self.logger.log(f"[INIT] Model: {getattr(config, 'model', 'N/A')}, Provider: {getattr(config, 'provider', 'N/A')}")
        self.logger.log(f"[INIT] Max threads: {getattr(config, 'max_threads', 'N/A')}")
        self.logger.log(f"[INIT] Unique labels: {self.unique_labels}")
        self.logger.log(f"[INIT] Aggregation keys: {self.aggregation_keys}")
        self.logger.log(f"[INIT] Answer extraction template: {self.ae_template}")
        
    def extract_answer(self, response):
        """
        Extracts the answer from the given response using a predefined template.

        Args:
            response (str): The response string to extract the answer from.
            ae_template (str): Path to the answer extraction template.
            config (dict): The configuration dictionary.

        Returns:
            dict: A dictionary containing the extracted answer.
        """        
        try:
            with open(self.ae_template, "r", encoding="utf-8") as f:
                template = f.read()
        except FileNotFoundError as e:
            self.logger.log(f"[EXTRACT_ANSWER] ❌ Template file not found: {self.ae_template} - {e}")
            return {"answer": "NA", "explanation": "Template file not found"}
        
        prompt = template.replace("{Response}", response)
        extracted_json = ""
        try:
            _, extracted_json, _ = llm.infer(prompt, max_tokens=1000, temperature=0, top_p=0.1, model=self.config.model, provider=self.config.provider)
            # Short-circuit when the upstream LLM returned its sentinel error
            # string — there's nothing useful to JSON-parse and the caller
            # needs a dict, not the raw error string. Mark with API_ERROR_LABEL
            # so the scoring pass can drop it from the accuracy denominator.
            if extracted_json in ("Could not get a response", "Refused by safety filter"):
                return {"answer": API_ERROR_LABEL, "explanation": extracted_json}
            extracted_json = utils.clean_llm_response(extracted_json)
            parsed_result = dirtyjson.loads(extracted_json)

            # Handle boolean values that might come from JSON parsing
            if isinstance(parsed_result, bool):
                # Convert boolean to string representation
                parsed_result = {"answer": "yes" if parsed_result else "no", "explanation": ""}
            elif isinstance(parsed_result, dict):
                # Check if any values in the dict are booleans and convert them
                if "answer" in parsed_result and isinstance(parsed_result["answer"], bool):
                    parsed_result["answer"] = "yes" if parsed_result["answer"] else "no"

            return parsed_result
        except ValueError as e:
            self.logger.log(f"[EXTRACT_ANSWER] ❌ JSON parsing failed: {e}")
            self.logger.log(f"[EXTRACT_ANSWER] Raw response that failed parsing: {extracted_json}")
            return {"answer": "NA", "explanation": f"JSON parse failed: {e}"}
        except Exception as e:
            self.logger.log(f"[EXTRACT_ANSWER] ❌ Unexpected error during LLM inference: {e}")
            return {"answer": "NA", "explanation": "LLM inference failed"}

    def get_llm_response(self, system_prompt, user_message, config, history=None, index=-1):
        """
        Retrieves a response from the language model.

        Args:
            system_prompt (str): The system prompt.
            user_message (str): The user message.
            config (dict): The configuration dictionary.
            history (list, optional): The conversation history. Defaults to None.
            index (int, optional): The index of the server. Defaults to -1.

        Returns:
            tuple: A tuple containing the response and token usage.
        """
        try:
            _, response, token_usage = llm.run_llm(
                user_message=user_message,
                system_prompt=system_prompt,
                history=history,
                model=config.model,
                provider=config.provider,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                index=index,
                reasoning_effort=getattr(config, "reasoning_effort", "minimal"),
                service_tier=getattr(config, "service_tier", "fast"),
            )
            return response, token_usage
        except Exception as e:
            self.logger.log(f"[LLM_RESPONSE] ❌ Error calling LLM: {e}")
            raise

    def populate_user_prompt_with_input(self, ex, user_prompt):
        """
        Populates the user prompt template with values from the example.
        Placeholders are defined as #KEY# (case-insensitive).

        Args:
            ex (dict): The example dictionary containing input values.
            user_prompt (str): The user prompt template.

        Returns:
            str: The user prompt with placeholders replaced by values from the example.
        """
        ex_id = ex.get('id', 'N/A')        
        if user_prompt and len(user_prompt) > 0:
            replacements_made = 0
            for key, value in ex.items():
                if key in ['id', 'truth', 'label', 'output']:
                    continue
                # Replace placeholders in both uppercase and lowercase formats.
                placeholder_upper = f"#{key.upper()}#"
                placeholder_lower = f"#{key}#"
                if placeholder_upper in user_prompt:
                    user_prompt = user_prompt.replace(placeholder_upper, str(value))
                    replacements_made += 1
                elif placeholder_lower in user_prompt:
                    user_prompt = user_prompt.replace(placeholder_lower, str(value))
                    replacements_made += 1
        else:
            user_prompt = ex.get('input', '')            
        return user_prompt

    def parse_response(self, response):
        """
        Parses the response from the language model to extract the answer and explanation.

        Args:
            response (str): The response string from the language model.
            extract_answer_func (function): The function to extract the answer from the response.
            unique_labels (list): A list of unique labels.

        Returns:
            tuple: A tuple containing the extracted answer and explanation.
        """        
        # Ensure response is a string to avoid 'bool' object has no attribute 'lower' error
        response_str = str(response) if response is not None else ""
        if "could not get a response" in response_str.lower() or response_str == "Refused by safety filter":
            self.logger.log(f"[PARSE_RESPONSE] ⚠️ API failure: {response_str[:60]}")
            return API_ERROR_LABEL, response_str

        answer_json = self.extract_answer(response)        
        try:
            if isinstance(answer_json, list):
                answer_json = answer_json[0]
                
            if isinstance(answer_json, str):
                answer = answer_json.upper()
                explanation = ""
            else:
                answer_value = answer_json.get("answer", "")
                answer = str(answer_value).upper() if answer_value else ""
                explanation = answer_json.get("explanation", "")
                
        except Exception as e:
            self.logger.log(f"[PARSE_RESPONSE] ❌ Error parsing response: {e}")
            self.logger.log(f"[PARSE_RESPONSE] Answer JSON that caused error: {answer_json}")
            self.logger.log(f"[PARSE_RESPONSE] Original response: {response[:500]}...")
            answer = response
            explanation = ""
        mapped_answer = self.map_answer_to_label(answer)
        # self.logger.log(f"Full Response: {response}, Parsed Answer: {answer}, Mapped Answer: {mapped_answer}, Parsed Explanation: {explanation}")
        return mapped_answer, explanation
        
    def map_answer_to_label(self, answer):
        """
        Maps the extracted answer to a canonical label.
        """
        # Convert answer to string to handle boolean and other types
        answer_str = str(answer) if answer is not None else ""
        for label in self.unique_labels:
            # Ensure label is string to prevent 'bool' object has no attribute 'lower' error
            label_str = str(label) if label is not None else ""
            if label_str.lower() in answer_str.lower():
                return label_str
        return "NA"
        
    def get_batch_score(self, labels, preds):
        scores = accuracy_score(labels, preds)
        return scores

    def get_instance_score(self, label, pred):
        return 1 if label == pred else 0
    
    def create_report(self, labels, preds):
        try:
            report = classification_report(labels, preds, output_dict=True)
            output_report = {}
            for key, metrics in report.items():
                if key in ["macro avg", "weighted avg", "accuracy"]:
                    continue
                output_report[key] = {
                    "precision": round(metrics["precision"], 3),
                    "recall": round(metrics["recall"], 3),
                    "f1score": round(metrics["f1-score"], 3),
                    "support": metrics["support"]  # support is typically an integer
                }
            output_report["macro_f1"] = round(report["macro avg"]["f1-score"], 3)
            output_report["weighted_f1"] = round(report["weighted avg"]["f1-score"], 3)
            output_report["accuracy"] = round(report["accuracy"], 3)
            
            self.logger.log(f"[CREATE_REPORT] ✅ Generated report - Accuracy: {output_report['accuracy']}, Macro F1: {output_report['macro_f1']}")
            return output_report
        except Exception as e:
            self.logger.log(f"[CREATE_REPORT] ❌ Error generating classification report: {e}")
            self.logger.log(f"[CREATE_REPORT] Labels sample: {labels[:10] if len(labels) > 10 else labels}")
            self.logger.log(f"[CREATE_REPORT] Predictions sample: {preds[:10] if len(preds) > 10 else preds}")
            self.logger.log(f"[CREATE_REPORT] Labels length: {len(labels)}, Predictions length: {len(preds)}")
            return {}

    def inference(self, ex, candidate, index=-1, max_servers=32):
        """
        Performs inference on a given example using a candidate prompt.

        Args:
            ex (dict): The example dictionary.
            candidate (Candidate): The candidate prompt.
            index (int, optional): The index of the server. Defaults to -1.
            max_servers (int, optional): The maximum number of servers. Defaults to 32.

        Returns:
            tuple: A tuple containing the example, prediction, explanation, response, and token usage.
        """
        ex_id = ex.get('id', 'N/A')
        system_prompt, conversation, user_message = candidate.get_textual_prompt()
        # Note: there used to be an in-memory `self.cache` lookup here, but it
        # was never written to (always missed) and just added a dict op per
        # inference. The disk-backed LLMCache covers the legitimate caching
        # path — leave it to that.
        i = index % max_servers
        user_message_populated = self.populate_user_prompt_with_input(ex, user_message)
        response, token_usage = self.get_llm_response(system_prompt, user_message_populated, self.config, conversation, index=i)
        prediction, explanation = self.parse_response(response)
        return ex, prediction, explanation, response, token_usage

    def run_evaluate(self, prompt, test_exs, desc):
        output_exs = []
        token_usage = utils.TokenUsage()
        total_prompt_tokens = 0
        count = 0
        errors_count = 0
        # M2: per-example retry on inference failure. The outer evaluate()
        # used to retry the WHOLE batch on BrokenProcessPool / SSLError,
        # which re-billed every example for one transient failure. Retry
        # only the failed examples (one extra attempt each).
        retried_indices: set[int] = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            futures = [executor.submit(self.inference, ex, prompt, i) for i, ex in enumerate(test_exs)]
            # Iterate in submission order (not as_completed) so output_exs is
            # built deterministically regardless of LLM-call latency variation.
            # Throughput is unchanged — future.result() blocks until that
            # specific future is done while other threads keep working.
            for orig_idx, future in enumerate(tqdm(futures, total=len(futures), desc=desc)):
                try:
                    ex, pred, explanation, response, token_usage_i = future.result()
                except Exception as e:
                    if orig_idx not in retried_indices:
                        retried_indices.add(orig_idx)
                        self.logger.log(
                            f"[RUN_EVALUATE] ⚠️ inference failed at idx={orig_idx} "
                            f"({type(e).__name__}: {e}); retrying once.")
                        try:
                            retry_fut = executor.submit(self.inference, test_exs[orig_idx], prompt, orig_idx)
                            ex, pred, explanation, response, token_usage_i = retry_fut.result()
                        except Exception as e2:
                            self.logger.log(
                                f"[RUN_EVALUATE] ❌ inference retry also failed at idx={orig_idx}: {e2}")
                            errors_count += 1
                            continue
                    else:
                        self.logger.log(f"[RUN_EVALUATE] ❌ Error processing future result: {e}")
                        errors_count += 1
                        continue
                ex_id = ex.get('id', 'N/A')
                true_label = ex.get('output', '')
                try:
                    instance_score = self.get_instance_score(true_label, pred)
                except Exception as e:
                    self.logger.log(f"[RUN_EVALUATE] ❌ Error calculating instance score for example {ex_id}: {e}")
                    instance_score = 0
                    errors_count += 1

                output_ex = ex.copy()
                output_ex.update({
                    'prediction': pred,
                    'response': response,
                    'explanation': explanation,
                    'score': instance_score
                })
                output_exs.append(output_ex)
                total_prompt_tokens += getattr(token_usage_i, 'prompt_tokens', 0)
                count += 1
                token_usage.update_token_usage(token_usage_i)
        
        # Calculate metrics. Drop API-error examples from the accuracy
        # denominator so codex timeouts / safety-filter refusals don't unfairly
        # penalize a candidate. They still appear in output_exs (with their
        # API_ERROR_LABEL prediction) for downstream inspection.
        scored_pairs = [(ex['output'], ex['prediction']) for ex in output_exs
                        if ex['prediction'] != API_ERROR_LABEL]
        api_error_count = len(output_exs) - len(scored_pairs)
        if api_error_count:
            self.logger.log(
                f"[RUN_EVALUATE] ⚠️ Excluded {api_error_count}/{len(output_exs)} "
                f"examples from score (API errors)")
        if scored_pairs:
            labels = [p[0] for p in scored_pairs]
            predictions = [p[1] for p in scored_pairs]
            batch_score = round(self.get_batch_score(labels, predictions), 3)
            report = self.create_report(labels, predictions)
        else:
            # All examples failed at the API layer — return 0 but flag clearly.
            self.logger.log("[RUN_EVALUATE] ❌ All examples were API errors; score=0")
            batch_score = 0.0
            report = {}
        avg_prompt_tokens = total_prompt_tokens / count if count else 0
        return batch_score, output_exs, report, token_usage, avg_prompt_tokens
    
    def evaluate(self, prompt, test_exs, output_path, desc="running evaluate"):
        retry_count = 0
        max_retries = 3
        
        while True:
            try:
                results = self.run_evaluate(prompt, test_exs, desc)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError) as e:
                retry_count += 1
                self.logger.log(f"[EVALUATE] ⚠️ Retry {retry_count}/{max_retries} due to error: {type(e).__name__}")
                if retry_count >= max_retries:
                    self.logger.log(f"[EVALUATE] ❌ Max retries reached, failing evaluation")
                    raise
                continue
            except Exception as e:
                self.logger.log(f"[EVALUATE] ❌ Unexpected error during evaluation: {e}")
                raise
        
        batch_score, output_exs, report, token_usage, avg_prompt_tokens = results
        self.save_inference(output_path, output_exs)
        return batch_score, output_exs, report, token_usage, avg_prompt_tokens

    def save_inference(self, output_path, output_exs):
        """
        Saves the inference results to a file in tab-separated format.

        Args:
            output_path (str): The path to the output file.
            output_exs (list): A list of dictionaries containing inference results.
        """
        if not output_exs:
            self.logger.log(f"[SAVE_INFERENCE] ⚠️ No examples to save to {output_path}")
            return
        file_exists = os.path.exists(output_path)
        mode = 'a' if file_exists else 'w'        
        try:
            with open(output_path, mode, encoding='utf-8') as outf:
                headers = list(output_exs[0].keys()) if output_exs else []                
                if mode == 'w':
                    outf.write("\t".join(headers) + "\n")
                saved_count = 0
                failed_count = 0
                
                for i, ex in enumerate(output_exs):
                    try:
                        cleaned_values = [str(ex.get(key, '')).strip().replace("\n", " ") for key in headers]
                        outf.write("\t".join(cleaned_values) + "\n")
                        saved_count += 1
                    except Exception as e:
                        self.logger.log(f"[SAVE_INFERENCE] ❌ Error saving example {i}: {e}")
                        failed_count += 1
                        continue            
        except Exception as e:
            self.logger.log(f"[SAVE_INFERENCE] ❌ Error opening/writing file {output_path}: {e}")
            raise

    def is_misclassified(self, input_text, true_label, predicted_label):
        """
        Return True if the predicted label does not match the true label.
        
        Args:
            input_text (str): The input text (unused here but kept for potential extension).
            true_label (any): The ground truth label.
            predicted_label (any): The predicted label.
        
        Returns:
            bool: True if there is a misclassification, else False.
        """
        return true_label != predicted_label

    def get_error_batch(self, output_exs):
        error_exs = [ex for ex in output_exs if self.is_misclassified(ex.get('input', ''), ex.get('output', ''), ex.get('prediction', ''))]
        error_rate = len(error_exs) / len(output_exs) * 100 if output_exs else 0
        return error_exs      

    def merge_reports(self, prompts, reports):        
        # Collect all flattened keys from candidate reports.
        all_keys = set()
        flat_reports = []
        for i, report in enumerate(reports):
            try:
                flat = utils.flatten_json(report)
                flat_reports.append(flat)
                all_keys.update(flat.keys())
            except Exception as e:
                self.logger.log(f"[MERGE_REPORTS] ❌ Error flattening report {i}: {e}")
                flat_reports.append({})
                
        sorted_metric_keys = sorted(all_keys)        
        rows = []
        for prompt, flat_report in zip(prompts, flat_reports):
            # Start the row with id from the prompt.
            row = {"id": getattr(prompt, "id", None)}
            # Add flattened report values (use None if key is missing).
            for key in sorted_metric_keys:
                row[key] = flat_report.get(key, None)
            # Add additional fields from the prompt if available.
            if hasattr(prompt, "get_info_dict"):
                try:
                    info = prompt.get_info_dict()
                    for k, v in info.items():
                        if k != "id":
                            row[k] = v
                except Exception as e:
                    self.logger.log(f"[MERGE_REPORTS] ⚠️ Error getting prompt info: {e}")
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Ensure 'id' is the first column.
        cols = list(df.columns)
        if "id" in cols:
            cols.remove("id")
            cols = ["id"] + cols
        df = df[cols]
        return df

    def aggregate_report_across_rounds(self, merged_test_report, round_number):
        aggregated = {}
        processed_keys = 0
        
        for key, values in merged_test_report.items():
            if key in self.aggregation_keys or (len(self.aggregation_keys)==0 and key != "id"):
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    avg_val = sum(numeric_values) / len(numeric_values)
                    max_val = max(numeric_values)
                    aggregated[f"{key}_avg"] = avg_val
                    aggregated[f"{key}_max"] = max_val
                    processed_keys += 1
                else:
                    self.logger.log(f"[AGGREGATE_REPORT] ⚠️ Skipping {key}: no numeric values found")
                    continue
                
        # Create a new row with the round number and aggregated metrics.
        row = {"round": round_number}
        row.update(aggregated)
    
        # If no global DataFrame is provided, create one.
        if self.global_df is None:
            self.global_df = pd.DataFrame([row])
        else:
            self.global_df = pd.concat([self.global_df, pd.DataFrame([row])], ignore_index=True)
        return self.global_df