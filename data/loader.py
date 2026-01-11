"""
Dataset loader module.

Provides unified interface for loading various datasets for evaluation.
Supports: COPA, StoryCloze, SST-2, SST-5, BoolQ, MMLU, XNLI, WinoGrande.
"""
import os
import json
import csv
import random
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


# Type alias for dataset format: (question, correct_answer, wrong_answer)
DataSample = Tuple[str, str, str]


class DatasetLoader:
    """
    Unified dataset loader supporting multiple benchmark datasets.
    
    Each dataset is loaded into a standard format:
    [(prompt, correct_answer, wrong_answer), ...]
    
    This format enables:
    - Direct use in activation intervention methods
    - Rollout-based pair generation
    - Consistent evaluation
    """
    
    def __init__(self, data_root: str = "./data"):
        """
        Initialize dataset loader.
        
        Args:
            data_root: Root directory containing dataset folders.
        """
        self.data_root = Path(data_root)
    
    def load(
        self, 
        dataset_name: str,
        split: str = "both",
        max_train: Optional[int] = None,
        max_test: Optional[int] = None
    ) -> Tuple[List[DataSample], List[DataSample]]:
        """
        Load a dataset by name.
        
        Args:
            dataset_name: Name of dataset ('copa', 'sst2', 'boolq', etc.).
            split: Which split to load ('train', 'test', 'both').
            max_train: Maximum training samples (None for all).
            max_test: Maximum test samples (None for all).
        
        Returns:
            Tuple of (train_data, test_data).
        """
        loaders = {
            "copa": self.load_copa,
            "storycloze": self.load_storycloze,
            "sst2": self.load_sst2,
            "sst5": self.load_sst5,
            "boolq": self.load_boolq,
            "mmlu": self.load_mmlu,
            "xnli": self.load_xnli,
            "winogrande": self.load_winogrande,
        }
        
        name = dataset_name.lower()
        if name not in loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
        
        train_data, test_data = loaders[name]()
        
        # Apply sample limits
        if max_train and len(train_data) > max_train:
            train_data = train_data[:max_train]
        if max_test and len(test_data) > max_test:
            test_data = test_data[:max_test]
        
        print(f"Loaded {dataset_name}: {len(train_data)} train, {len(test_data)} test samples")
        
        return train_data, test_data
    
    def load_copa(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load COPA (Choice of Plausible Alternatives) dataset."""
        train_data, test_data = [], []
        
        # Load training data
        train_path = self.data_root / "xcopa" / "train.csv"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[1:1501]  # Skip header, limit samples
            
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                label, premise, question = parts[0].strip(), parts[2].strip(), parts[3].strip()
                choice1, choice2 = parts[4].strip(), parts[5].strip()
                
                prompt = (
                    f"Question:\n{premise} Based on the previous passage, "
                    f"choose the most reasonable {question}.\n"
                    f"A:{choice1}\nB:{choice2}\n\nAnswer:\n"
                )
                
                if int(label) == 0:
                    train_data.append((prompt, "A", "B"))
                else:
                    train_data.append((prompt, "B", "A"))
        
        # Load test data
        test_path = self.data_root / "xcopa" / "test.en.jsonl"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    premise = item['premise']
                    question = item['question']
                    choice1, choice2 = item['choice1'], item['choice2']
                    label = item['label']
                    
                    prompt = (
                        f"Question:\n{premise} Based on the previous passage, "
                        f"choose the most reasonable {question}.\n"
                        f"A:{choice1}\nB:{choice2}\n\nAnswer:\n"
                    )
                    
                    if int(label) == 0:
                        test_data.append((prompt, "A", "B"))
                    else:
                        test_data.append((prompt, "B", "A"))
        
        return train_data, test_data
    
    def load_storycloze(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load StoryCloze dataset."""
        train_data, test_data = [], []
        
        def process_line(line: str) -> Optional[DataSample]:
            parts = line.strip().split('\t')
            if len(parts) < 8:
                return None
            
            sent1, sent2, sent3, sent4 = parts[1], parts[2], parts[3], parts[4]
            quiz1, quiz2, label = parts[5], parts[6], parts[7]
            
            context = f"{sent1} {sent2} {sent3} {sent4}"
            prompt = (
                f"{context}\nQuestion: What is a possible continuation for the story "
                f"given the following options?\nA: {quiz1} B: {quiz2}\nAnswer:"
            )
            
            if int(label) == 1:
                return (prompt, "A", "B")
            else:
                return (prompt, "B", "A")
        
        # Load training data
        train_path = self.data_root / "xstorycloze" / "spring2016.val.en.tsv.split_20_80_train.tsv"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:  # Skip header
                    sample = process_line(line)
                    if sample:
                        train_data.append(sample)
        
        # Load test data
        test_path = self.data_root / "xstorycloze" / "spring2016.val.en.tsv.split_20_80_eval.tsv"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    sample = process_line(line)
                    if sample:
                        test_data.append(sample)
        
        return train_data, test_data
    
    def load_sst2(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load SST-2 (Stanford Sentiment Treebank - Binary) dataset."""
        train_data, test_data = [], []
        
        def process_item(item: Dict) -> DataSample:
            text = item['text'].strip()
            label = item['label_text'].strip()
            
            prompt = (
                f"Consider the sentiment expression in this sentence and respond briefly "
                f"with 'positive' or 'negative'.\n\n{text}\n\nAnswer:"
            )
            
            if label == 'positive':
                return (prompt, "Positive", "Negative")
            else:
                return (prompt, "Negative", "Positive")
        
        # Load training data
        train_path = self.data_root / "SST" / "sst2" / "train.jsonl"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 2000:  # Limit training samples
                        break
                    train_data.append(process_item(json.loads(line)))
        
        # Load test data
        test_path = self.data_root / "SST" / "sst2" / "test.jsonl"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    test_data.append(process_item(json.loads(line)))
        
        return train_data, test_data
    
    def load_sst5(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load SST-5 (5-class sentiment) dataset."""
        train_data, test_data = [], []
        
        opposite_map = {
            'positive': 'negative',
            'very positive': 'very negative',
            'negative': 'positive',
            'very negative': 'very positive',
            'neutral': None  # Random choice
        }
        
        def process_item(item: Dict) -> DataSample:
            text = item['text'].strip()
            label = item['label_text'].strip()
            
            prompt = (
                f"Consider the sentiment expression in this sentence and respond briefly "
                f"with 'very positive', 'positive', 'neutral', 'negative', or 'very negative'.\n\n"
                f"{text}\n\nAnswer:"
            )
            
            wrong = opposite_map.get(label)
            if wrong is None:
                wrong = random.choice(['positive', 'negative'])
            
            return (prompt, label, wrong)
        
        # Load training data
        train_path = self.data_root / "SST" / "sst5" / "train.jsonl"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 2000:
                        break
                    train_data.append(process_item(json.loads(line)))
        
        # Load test data
        test_path = self.data_root / "SST" / "sst5" / "test.jsonl"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    test_data.append(process_item(json.loads(line)))
        
        return train_data, test_data
    
    def load_boolq(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load BoolQ (Boolean Questions) dataset."""
        train_data, test_data = [], []
        
        def process_item(item: Dict) -> DataSample:
            question = item['question'].strip()
            passage = item['passage'].strip()
            answer = str(item['answer']).strip()
            
            prompt = (
                f"Is the answer to the question encapsulated in the passage? "
                f"Please confirm with 'yes' or 'no'.\n\n"
                f"Passage: {passage}\n\nQuestion: {question}\n\nAnswer:"
            )
            
            if answer == 'True':
                return (prompt, "Yes", "No")
            else:
                return (prompt, "No", "Yes")
        
        # Load training data
        train_path = self.data_root / "boolq" / "train.jsonl"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 2000:
                        break
                    train_data.append(process_item(json.loads(line)))
        
        # Load test data (dev set)
        test_path = self.data_root / "boolq" / "dev.jsonl"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    test_data.append(process_item(json.loads(line)))
        
        return train_data, test_data
    
    def load_mmlu(self, subject: str = "all") -> Tuple[List[DataSample], List[DataSample]]:
        """Load MMLU (Massive Multitask Language Understanding) dataset."""
        train_data, test_data = [], []
        
        mmlu_dir = self.data_root / "mmlu" / "test"
        if not mmlu_dir.exists():
            return train_data, test_data
        
        def process_row(row: List[str]) -> Optional[DataSample]:
            if len(row) < 6:
                return None
            
            question = row[0]
            options = row[1:5]
            correct = row[5]
            
            prompt = (
                f"Question: {question}\n"
                f"Which of the following answers is correct?\n"
                f"A. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n"
                f"State the letter corresponding to the correct answer.\nAnswer:"
            )
            
            wrong_choices = ['A', 'B', 'C', 'D']
            wrong_choices.remove(correct)
            wrong = random.choice(wrong_choices)
            
            return (prompt, correct, wrong)
        
        # Load from all subject files or specific subject
        for csv_file in mmlu_dir.glob("*.csv"):
            if subject != "all" and subject.lower() not in csv_file.stem.lower():
                continue
            
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= 500:  # Limit per subject
                        break
                    sample = process_row(row)
                    if sample:
                        train_data.append(sample)
        
        # Use same data for test (MMLU doesn't have separate train/test in this format)
        test_data = train_data.copy()
        
        return train_data, test_data
    
    def load_xnli(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load XNLI (Cross-lingual NLI) dataset - English subset."""
        train_data, test_data = [], []
        
        label_map = {
            'entailment': ('True', ['False', 'Inconclusive']),
            'contradiction': ('False', ['True', 'Inconclusive']),
            'neutral': ('Inconclusive', ['True', 'False'])
        }
        
        def process_line(parts: List[str]) -> Optional[DataSample]:
            if len(parts) < 8:
                return None
            
            lang, label = parts[0], parts[1]
            premise, hypothesis = parts[6], parts[7]
            
            if lang != 'en':
                return None
            
            prompt = (
                f"Answer whether the hypothesis is more likely to be true, false, "
                f"or inconclusive based on the given premise.\n"
                f"Premise: {premise}\nHypothesis: {hypothesis}\nAnswer:"
            )
            
            if label in label_map:
                correct, wrong_options = label_map[label]
                wrong = random.choice(wrong_options)
                return (prompt, correct, wrong)
            return None
        
        # Load dev data as train
        dev_path = self.data_root / "mnli" / "xnli.dev.tsv"
        if dev_path.exists():
            with open(dev_path, "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:  # Skip header
                    sample = process_line(line.split('\t'))
                    if sample:
                        train_data.append(sample)
        
        # Load test data
        test_path = self.data_root / "mnli" / "xnli.test.tsv"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    sample = process_line(line.split('\t'))
                    if sample:
                        test_data.append(sample)
        
        return train_data, test_data
    
    def load_winogrande(self) -> Tuple[List[DataSample], List[DataSample]]:
        """Load WinoGrande (commonsense reasoning) dataset."""
        train_data, test_data = [], []
        
        def process_item(item: Dict) -> DataSample:
            sentence = item['sentence'].strip()
            option1 = item['option1'].strip()
            option2 = item['option2'].strip()
            answer = item['answer'].strip()
            
            prompt = (
                f"Please fill in the blanks. Write A or B as the answer.\n\n"
                f"Sentence: {sentence}\nA. {option1}\nB. {option2}\nAnswer:"
            )
            
            if answer == '1':
                return (prompt, "A", "B")
            else:
                return (prompt, "B", "A")
        
        # Load training data
        train_path = self.data_root / "winogrande" / "train_m.jsonl"
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 2000:
                        break
                    train_data.append(process_item(json.loads(line)))
        
        # Load dev data as test
        test_path = self.data_root / "winogrande" / "dev.jsonl"
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    test_data.append(process_item(json.loads(line)))
        
        return train_data, test_data
