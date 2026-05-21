from src.base_task import BaseTask
import pandas as pd

class Task(BaseTask):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_examples(self, path, ex_type):
        df = pd.read_csv(path, sep="\t", header=None, encoding="utf-8")
        df.columns = ["input", "output", "single_label"]

        exs = []
        for k, row in df.iterrows():
            output_val = str(row["output"])
            ex = {
                "id": f"{ex_type}-{k}",
                "input": row["input"],
                "output": output_val
            }
            exs.append(ex)
        return exs