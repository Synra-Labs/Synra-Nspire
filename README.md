# Synra Nspire


Synra Labs™ is an AI research and development lab on a mission to democratize advanced model creation and drive the world toward autonomous intelligence.  

Our flagship product, **Synra Nspire**, is a platform for **custom LLM generation, training, and deployment**. It enables individuals and enterprises to build, fine-tune, and manage domain-specific language models with minimal overhead — leveraging **AI Logic** to automate much of the engineering effort traditionally required.

---

##  Why Synra Nspire?

Building a useful large language model (LLM) today requires:
- Access to expensive compute clusters.
- Teams of ML engineers to configure fine-tuning pipelines.
- Domain experts to carefully prepare data.

This process can take months and cost hundreds of thousands of dollars.

**Synra Nspire eliminates this bottleneck** by providing:
- **Pre-curated open-source base models** (optimized and distilled).  
- **Automated fine-tuning pipelines** for domain adaptation.  
- **Data upload & management** via secure S3 storage.  
- **Evaluation and benchmarking tools** to compare and improve models.  
- **Deployment-ready artifacts** exportable for local, on-prem, or cloud use.  

From **finance compliance assistants** to **biotech research copilots**, Nspire shortens the distance between an idea and a running domain-specific LLM.

---

##  Features

###  Model Customization
- Select from **six top-tier open-source models** (Mistral-7B, LLaMA-3-8B, Phi-3-Mini, Qwen2-1.5B, Gemma-2B, TinyLlama).  
- Adjust runtime parameters (temperature, token limit, sampling strategies).  
- Full JSON override for power users (training recipes, optimizers, quantization).

###  Secure Data Integration
- Upload datasets (CSV, JSON, TXT, or image files) via **signed S3 URLs**.  
- Automatic schema detection and validation.  
- Support for **structured text, tabular data, and multimodal datasets** (images for medical, logistics, etc.).

###  Training Pipelines
- Orchestrated fine-tuning with **LoRA/QLoRA** for parameter-efficient training.  
- Distributed training on **AWS EC2 GPU instances** or **EKS with GPU node pools**.  
- Training jobs queued and monitored via **job orchestrator** (Celery/Redis or SQS).  
- Supports:
  - Supervised Fine-Tuning (SFT).  
  - Reinforcement Learning from Human Feedback (RLHF/DPO).  
  - Domain-specific adapters and merges.  

###  Evaluation & Benchmarking
- Built-in evaluation suites:
  - Perplexity, Rouge/BLEU, accuracy (text).  
  - Top-k metrics (classification).  
  - Domain-specific probes (e.g., compliance rules).  
- Automated comparison between baseline vs fine-tuned models.  
- Human evaluation hooks for qualitative scoring.

###  Deployment
- Export options:
  - **Adapter-only “Brain” files** (LoRA modules + configs).  
  - **Merged weights** (if license permits).  
  - **ONNX/TensorRT** formats for optimized inference.  
- Deploy models to:
  - Local machines (inference with `transformers` or `llama.cpp`).  
  - AWS SageMaker endpoints.  
  - Containerized inference servers (Docker + Triton).  

###  Continuous Improvement
- AI-assisted modification engine: suggests better configs based on dataset analysis and previous training runs.  
- Iterative “Improve Model” button that spawns new training jobs with refined parameters.

---

## 🛠 Architecture Overview

### High-Level Components
1. **Frontend (Web Portal)**  
   - React + Tailwind for user interface.  
   - Auth system (Cognito/Identity Provider).  
   - Model selection, configuration, job monitoring.

2. **Backend (API & Orchestration)**  
   - Python FastAPI serving REST/GraphQL APIs.  
   - PostgreSQL (metadata & job tracking).  
   - Redis or Amazon SQS for job queues.  

3. **Storage**  
   - **S3 bucket** for datasets, training artifacts, model weights.  
   - Versioned storage for reproducibility.  
   - KMS encryption + signed URL access.

4. **Training Cluster**  
   - AWS EKS with GPU node groups.  
   - Workers pull jobs from queue and launch training with:
     - Hugging Face Transformers/Datasets.  
     - PEFT (LoRA/QLoRA).  
     - Accelerate/DeepSpeed.  

5. **Model Registry**  
   - Metadata of each trained model stored in Postgres.  
   - Artifacts (adapters, configs, logs, eval results) stored in S3.  
   - Model lineage tracked (base → fine-tuned → merged).

6. **Evaluation Service**  
   - Periodic eval jobs run on validation sets.  
   - Reports stored alongside model artifacts.  
   - User-facing dashboards to compare results.

7. **Deployment Layer**  
   - SageMaker endpoints or containerized inference servers.  
   - CI/CD pipelines to package updated models.

---

##  Example Project Workflow

1. **Create Project**  
   - Login and create a new project.  
   - Select base model (e.g., Mistral-7B).  

2. **Upload Data**  
   - Drag-and-drop CSV files or medical images.  
   - Data is uploaded securely to S3.  

3. **Configure Training**  
   - Choose epochs, batch size, optimizer, quantization.  
   - Optionally use Advanced JSON overrides.  

4. **Launch Training Job**  
   - Backend schedules training on AWS GPU cluster.  
   - Job status visible on dashboard.  

5. **Evaluate Results**  
   - Auto-generated metrics & evaluation reports.  
   - Compare against previous runs.  

6. **Download or Deploy**  
   - Export Brain JSON, LoRA adapters, or merged weights.  
   - Deploy to SageMaker or download for local inference.

---

##  Brain JSON (Infrastructure-Ready)

Example Brain file that ties into S3 + AWS orchestration:

```json
{
  "version": "0.2",
  "created_at": "2025-08-24T18:00:00Z",
  "project": {
    "name": "Finance Compliance Assistant",
    "id": "proj-12345"
  },
  "base_model": {
    "name": "mistral-7b-instruct",
    "family": "mistral",
    "parameters_b": 7,
    "context_window": 8192,
    "license": "Apache-2.0"
  },
  "runtime": {
    "temperature": 0.2,
    "max_tokens": 2048
  },
  "instruction_profile": {
    "system_prompt": "You are an AI assistant trained for financial compliance.",
    "user_additional_instructions": "Always cite relevant regulations."
  },
  "training_recipe": {
    "method": "LoRA",
    "quantization": "4-bit",
    "epochs": 5,
    "learning_rate": 0.0002,
    "batch_size": 16
  },
  "data_sources": [
    {
      "bucket": "synra-nspire-datasets",
      "path": "finance/compliance_rules.csv",
      "type": "csv"
    }
  ],
  "artifacts": {
    "bucket": "synra-nspire-models",
    "adapters": "proj-12345/lora-ep5",
    "merged": "proj-12345/merged-weights",
    "reports": "proj-12345/eval-v1.json"
  }
}
