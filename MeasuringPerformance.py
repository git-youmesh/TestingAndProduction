from transformers import pipeline
from datasets import load_dataset
import evaluate
import torch
from pathlib import Path
from time import perf_counter
from time import perf_counter
metric = evaluate.load("accuracy")
accuracy_score = metric
import numpy as np

def time_pipeline(self, query="What is the pin number for my account?"):
 """This overrides the PerformanceBenchmark.time_pipeline()
method"""
 latencies = []
 # Warmup
 for _ in range(10):
    _ = self.pipeline(query)
  # Timed run
 for _ in range(100):
  start_time = perf_counter()
  _ = self.pipeline(query)
  latency = perf_counter() - start_time
  latencies.append(latency)
  # Compute run statistics
  time_avg_ms = 1000 * np.mean(latencies)
  time_std_ms = 1000 * np.std(latencies)
 print(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
 return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}



def compute_size(self):
    """This overrides the PerformanceBenchmark.compute_size()
    method"""
    state_dict = self.pipeline.model.state_dict()
    tmp_path = Path("model.pt")
    torch.save(state_dict, tmp_path)
    # Calculate size in megabytes
    size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
    # Delete temporary file
    tmp_path.unlink()
    print(f"Model size (MB) - {size_mb:.2f}")
    return {"size_mb": size_mb}

class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, optim_type="BERTbaseline"):
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type
    def compute_accuracy(self):
    # We'll define this later
      pass
    def compute_size(self):
    # We'll define this later
      pass
    def time_pipeline(self):
    # We'll define this later
      pass
    def run_benchmark(self):
      metrics = {}
      metrics[self.optim_type] = self.compute_size()
      metrics[self.optim_type].update(self.time_pipeline())
      metrics[self.optim_type].update(self.compute_accuracy())
      return metrics



clinc = load_dataset("clinc_oos", "plus")

bert_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
pipe = pipeline("text-classification", model=bert_ckpt)

query = """Hey, I'd like to rent a vehicle from Nov 1st to Nov 15th
in
Paris and I need a 15 passenger van"""
 

sample = clinc["test"][42]
intents = clinc["test"].features["intent"]

def compute_accuracy(self):
    """This overrides the PerformanceBenchmark.compute_accuracy()
    method"""
    preds, labels = [], []
    for example in self.dataset:
        pred = self.pipeline(example["text"])[0]["label"]
        label = example["intent"]
        preds.append(intents.str2int(pred))
        labels.append(label)
    accuracy = accuracy_score.compute(predictions=preds,references=labels)
    print(f"Accuracy on test set - {accuracy['accuracy']:.3f}")
    return accuracy
PerformanceBenchmark.compute_accuracy = compute_accuracy
list(pipe.model.state_dict().items())[42]
torch.save(pipe.model.state_dict(), "model.pt")

PerformanceBenchmark.compute_size = compute_size
# Time varies for each execution for same Query. To get accuate latency 
# query is performced many times , distrubution is anlyzed by taking mean 
# and standerd deviation of latency tme 
for _ in range(3):
    start_time = perf_counter()
    _ = pipe(query)
    latency = perf_counter() - start_time
    print(f"Latency (ms) - {1000 * latency:.3f}")

PerformanceBenchmark.time_pipeline = time_pipeline
