from src.base_task import BaseTask
import pandas as pd

class Task(BaseTask):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.unique_labels = ['A', 'B', 'C', 'D']   # List to store unique labels

    def get_examples(self, path, ex_type):
        df = pd.read_csv(path, sep="\t", header=None, encoding="utf-8")
        df.columns = ["input", "output"]

        exs = []
        for k, row in df.iterrows():
            output_values = str(row["output"])
            for label in self.unique_labels:
                if label in output_values:
                    output_values = label
                    break
            ex = {
                "id": f"{ex_type}-{k}",
                "input": row["input"],
                "output": output_values
            }
            exs.append(ex)
        return exs
    
    def input_to_str(self, ex):
        """
        Convert the input values of an example to a string.
        """
        return str(ex["input"])