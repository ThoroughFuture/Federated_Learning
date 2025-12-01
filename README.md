# Federated Learning Guide

## 📚 Model Principle

### What is Federated Learning?

**Federated Learning** is a distributed machine-learning paradigm that lets many clients (e.g., hospitals, companies) jointly train a single global model without ever sharing their raw data.

### Core Characteristics

1. **Data never leaves the premises** – each client keeps its dataset locally  
2. **Only model weights are shared** – raw data are never transmitted  
3. **Privacy preservation** – institutional data privacy is protected  
4. **Collaborative training** – multiple organizations can co-train a stronger model

### FedAvg Algorithm

**FedAvg (Federated Averaging)** is the canonical federated-learning algorithm:

1. **Initialization** – server initializes global model w₀  
2. **Client training** – every client k trains locally for E epochs  
   - Client k holds nₖ samples  
   - Produces local model wₖ  
3. **Server aggregation** – server aggregates the local weights  

   $$
   w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_k
   $$

   where K is the number of clients, nₖ the sample count at client k, and n = Σ nₖ the total sample count.  
4. **Broadcast** – server sends the updated global model w_{t+1} back to every client  
5. **Repeat** – steps 2–4 iterate until convergence


---

## 🔧 Usage

### Quick Start

1. **Edit config**:

   ```python
   device = 'cuda:0'
   
   clients_data = {
       0: load_data_from_pos_neg(
           pos_dir='../data/pos_image.txt',
           neg_dir='../data/neg_image.txt'
       ),
   }

  2. **Launch training**:
  
  federated_learning.py
