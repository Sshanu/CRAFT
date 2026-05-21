import os
import pandas as pd
import random
import numpy as np

class BaseTask:
    def __init__(self, data_dir, random_seed=42):
        self.data_dir = data_dir
        self.label_mapping = {}   # Maps a label (string) to a numeric value
        self.unique_labels = []   # List to store unique labels
        # Create a seeded random generator for reproducible sampling
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed

    def update_unique_labels(self, label):
        """
        Update the list of unique labels with the given label if not already present,
        and update the label mapping to map the label to a unique number.
        """
        if label not in self.unique_labels:
            self.unique_labels.append(label)
            # Map the label to its index (a unique number)
            self.label_mapping[label] = len(self.unique_labels) - 1

    def get_unique_labels(self):
        """
        Return the list of unique labels.
        """
        return self.unique_labels

    def get_examples(self, path, ex_type):
        df = pd.read_csv(path, sep="\t", header=None, encoding="utf-8")    
        exs = []
    
        # Iterate over DataFrame rows to build examples and update unique labels.
        for k, row in df.iterrows():
            output_val = str(row["output"])
            # Update unique labels and mapping.
            self.update_unique_labels(output_val)
            ex = {
                "id": f"{ex_type}-{k}",
                "input": row["input"],
                "output": output_val
            }
            exs.append(ex)
        
        return exs
        
    def reseed(self, new_seed):
        """Reseed the random number generator for this task."""
        self.random_seed = new_seed
        self.rng = np.random.default_rng(new_seed)

    def get_train_examples(self):
        return self.get_examples(os.path.join(self.data_dir, "train.tsv"), "train")
        
    def get_validation_examples(self):
        validation_path = os.path.join(self.data_dir, "validation.tsv")
        if os.path.isfile(validation_path):
            validation_exs = self.get_examples(validation_path, "validation")
        else:
            validation_exs = self.get_examples(os.path.join(self.data_dir, "train.tsv"), "validation")
        return validation_exs

    def get_test_examples(self):
        return self.get_examples(os.path.join(self.data_dir, "test.tsv"), "test")

    def stringify_prediction(self, pred):
        """
        Given a numeric prediction, return its corresponding label string.
        """
        # Reverse lookup in label_mapping
        for label, number in self.label_mapping.items():
            if number == pred:
                return label
        return "NA"

    def data_sampler(self, examples, num_samples, mode="random"):
        """
        Sample a set of examples based on the specified mode using seeded randomness.

        Args:
            examples (list): A list of example dictionaries.
            num_samples (int): Total number of samples to return.
            mode (str): Sampling mode - "random" returns a random sample from the dataset,
                        "classwise" returns a class-balanced sample using self.unique_labels.

        Returns:
            list: A list of sampled examples.
        """
        if mode == "random":
            # Use numpy's random generator for reproducible sampling
            if num_samples >= len(examples):
                return examples[:]
            indices = self.rng.choice(len(examples), size=min(num_samples, len(examples)), replace=False)
            return [examples[i] for i in indices]
        elif mode == "classwise":
            # Use self.unique_labels to define the classes.
            classes = self.unique_labels
            num_classes = len(classes)
            if num_classes == 0:
                return []
            samples_per_class = num_samples // num_classes
            
            # Group examples by label (assuming label is stored in the "output" field).
            examples_by_label = {}
            for ex in examples:
                label = ex["output"]
                examples_by_label.setdefault(label, []).append(ex)
            
            sampled_examples = []
            # For each class, randomly sample the desired number of examples using seeded RNG.
            for cl in classes:
                if cl in examples_by_label:
                    available = examples_by_label[cl]
                    n_samples = min(samples_per_class, len(available))
                    if n_samples > 0:
                        if n_samples >= len(available):
                            sampled_examples.extend(available)
                        else:
                            indices = self.rng.choice(len(available), size=n_samples, replace=False)
                            sampled_examples.extend([available[i] for i in indices])
            
            # If we have fewer samples than desired, fill in with random examples.
            if len(sampled_examples) < num_samples:
                remaining = num_samples - len(sampled_examples)
                # Get examples not already selected
                selected_ids = {ex['id'] for ex in sampled_examples}
                available_examples = [ex for ex in examples if ex['id'] not in selected_ids]
                
                if available_examples and remaining > 0:
                    if remaining >= len(available_examples):
                        sampled_examples.extend(available_examples)
                    else:
                        indices = self.rng.choice(len(available_examples), size=remaining, replace=False)
                        sampled_examples.extend([available_examples[i] for i in indices])
            return sampled_examples
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")        