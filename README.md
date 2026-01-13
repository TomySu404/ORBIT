# ORBIT: On-distribution Rollout-based Behavioral Intervention Technique

ORBIT is a novel activation steering framework designed to improve the behavioral alignment and performance of Large Language Models (LLMs) through on-distribution intervention. It moves beyond traditional contrastive activation addition (CAA) by introducing rollout-based pair generation and continuous soft scaling.

## 🚀 Key Features

- **ORC (On-distribution Rollout-based Contrastive pair generation)**: Reduces training-inference distribution shift by using model-generated rollouts.
- **CSS (Continuous Soft Scaling)**: Replaces discrete binary masking with continuous scaling weights to preserve steering direction.
- **Re-read Fallback**: Robust handling of samples where the model fails to generate correct responses naturally.

---

## 🧠 Algorithm Principles

### 1. On-distribution Rollout-based Contrastive Pair Generation (ORC)

Traditional methods often rely on manually constructed pairs like `(Question + Correct Answer)` vs `(Question + Wrong Answer)`. However, these "teacher-forced" activations differ significantly from the model's natural "auto-regressive" activation distribution during inference.

ORBIT addresses this via **Rollout**:
1. For a given question $q$, we sample $n$ responses $\{r_1, r_2, \dots, r_n\}$ using temperature sampling.
2. Each response is classified as **Correct** ($r^+$) or **Incorrect** ($r^-$) based on the ground truth.
3. We form all possible contrastive pairs $(r^+, r^-)$ from these rollouts.

**Mathematical Advantage**:
- **Distribution Alignment**: Our estimate $\mathbb{E}[h|q, r^+] - \mathbb{E}[h|q, r^-]$ is computed under the model's actual generation distribution $p_{ar}$, eliminating the training-inference gap.
- **Variance Reduction**: If a sample yields $k$ correct and $m$ incorrect responses, we obtain $k \times m$ pairs, significantly reducing the variance of the estimated steering vector.

### 2. Continuous Soft Scaling (CSS)

Standard steering methods often use **Top-K masking**, where only the top $k$ dimensions with the largest differences are kept (binary mask $\in \{0, 1\}^d$). This can distort the semantic direction of the steering vector.

ORBIT introduces **Continuous Scaling**:
Instead of a binary mask, we compute a continuous weight vector $\beta$ $\in [0, 1]^d$. The intervention formula is:

 h' = h + $\alpha$ $\cdot \beta \odot \mu$

where:
- $h$ is the original activation.
- $\alpha$ is the intervention strength.
- $\mu$ is the mean difference vector.
- $\beta$ is the scaling weight derived from $\mu$.

**Scaling Methods**:
- **Max Norm**: $\beta_i = \Delta h_i / \max(|\Delta h|)$. Preserves relative magnitudes while bounding values.
- **L2 Norm**: $\beta_i = \Delta h_i / \|\Delta h\|_2$. Ensures the steering vector is a unit vector, preserving direction exactly.
- **Softmax**: $\beta_i = \text{softmax}(|\Delta h|)_i \cdot \text{sign}(\Delta h_i) \cdot d$. Strongly emphasizes dimensions with the most significant differences.

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/username/ORBIT.git
cd ORBIT

# Install dependencies (using Tsinghua source as per requirements)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 💻 Usage

### Basic Intervention
```bash
python main.py --model meta-llama/Llama-3.1-8B-Instruct --dataset copa --strength 1.2
```

### Advanced Configuration
```bash
python main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num_rollouts 8 \
    --scaling softmax \
    --layer_scope last_n \
    --num_layers 10 \
    --reread_weight 0.3
```

---

## 📂 Project Structure

- `main.py`: Entry point for experiments.
- `steering/`: Core implementation of ORBIT algorithms.
    - `rollout.py`: ORC pair generation logic.
    - `diff_vector.py`: CSS weight computation and aggregation.
    - `intervention.py`: Hook-based activation intervention.
- `models/`: Model wrappers and activation extraction.
- `data/`: Dataset loaders and preprocessing.
- `utils/`: Metrics and evaluation helpers.

---

## 📄 License
This project is licensed under the MIT License.

