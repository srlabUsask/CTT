# Carbon-Taxed Transformers Replication Package

This repository provides a comprehensive replication package for the study titled *Carbon-Taxed Transformers: A Green Compression Pipeline for Overgrown Language Models*. It includes neural architecture search (NAS) scripts, training/evaluation pipelines, and empirical analysis figures for three tasks: code clone detection, code generation, and code summarization.

## Repository Structure

The repository is organized into the following directories:

### 1. `clone_detection/`

This folder contains code and data for conducting NAS and training models for the **code clone detection** task.

- **`code_clone_nas.py`**: Entry point to run NAS. Execute via:
  ```
  python code_clone_nas.py
  ```

- **Subdirectories:**
  - **`compress/`**: Contains scripts for training and evaluating student models. Execution instructions are provided within the folder.
  - **`data/`**: Contains all data related to code clone detection.
  - **`finetune/`**: Contains scripts for training and evaluating teacher models. Instructions are included within the folder.

### 2. `code_generation/`

This folder contains code and training pipelines for the **code generation** task.

- **`code_gen_nas.py`**: Entry point to run NAS for code generation. Execute via:
  ```
  python code_gen_nas.py
  ```

- **Subdirectories:**
  - **`compress/`**: Scripts for student model training and evaluation.
  - **`finetune/`**: Scripts for teacher model training and evaluation.

Execution instructions are available inside the respective folders.

### 3. `code_summarization/`

This folder provides code and utilities for the **code summarization** task.

- **`code_sum_nas.py`**: Entry point to run NAS for code summarization. Execute via:
  ```
  python code_sum_nas.py
  ```

- **Subdirectories:**
  - **`compress/`**: Scripts for student model training and evaluation.
  - **`data/`**: All task-related data.
  - **`finetune/`**: Scripts for teacher model training and evaluation.

Detailed usage instructions are provided within the respective subfolders.

### 5. `requirements.txt`

This file lists the Python package dependencies required to run the code in this repository. Install them using:

```
pip install -r requirements.txt
```

