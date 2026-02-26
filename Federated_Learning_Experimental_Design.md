
# 🧪 Federated Learning Experimental Design

## 1. Experimental Dimensions

We define four core dimensions that fully describe each experiment configuration.

### Dim 1 — Framework

| Code | Framework |
|------|-----------|
| Dim_1_1 | Flower |
| Dim_1_2 | FEDn |
| Dim_1_3 | FedML |

---

### Dim 2 — Strategy

| Code | Strategy |
|------|----------|
| Dim_2_1 | FedAvg |
| Dim_2_2 | FedYogi |
| Dim_2_3 | FedAdam |

---

### Dim 3 — Dataset (IID Level)

| Code | Description |
|------|------------|
| Dim_3_1 | Dataset_100 (100% IID) |
| Dim_3_2 | Dataset_050 (50% IID) |
| Dim_3_3 | Dataset_000 (0% IID – fully non-IID) |

---

### Dim 4 — Failing Clients

| Code | Failure % |
|------|----------|
| Dim_4_1 | 25% |
| Dim_4_2 | 50% |
| Dim_4_3 | 75% |

---

# 2. Experiment Execution Logic

Each experiment is generated using nested loops over all dimensions:

```python
for d1 in Dim_1:
    for d2 in Dim_2:
        for d3 in Dim_3:
            for d4 in Dim_4:
                RUN_ID = 11213141
                START: RUN_ID
```

Each experiment is defined by:

Framework + Strategy + Dataset + FailingClients

---

# 3. Experiment Naming Convention

Each experiment is named as:

EXP_<RUN_ID>

Example:

EXP_11213141

The digits encode:

Dim_1_1 + Dim_2_1 + Dim_3_1 + Dim_4_1

---

# 4. Experiment Outputs

Each experiment produces:

- `.pd` file → Final trained model  
- `.json` file → Logs and metrics  

After each experiment:

- A Python script is executed  
- It generates plots and summary statistics  

---

# 5. Experiment Progress Tracking Table

Legend:
- ⬜ Not started
- 🟡 Running
- ✅ Completed
- ❌ Failed

| Strategy | Dataset | Fail % | Flower | FEDn | FedML |
|-----------|----------|--------|--------|------|--------|
| FedAvg | 100IID | 25% | ✅ | ⬜ | ⬜ |
| FedAvg | 100IID | 50% | 🟡 | ⬜ | ⬜ |
| FedAvg | 100IID | 75% | ⬜ | ⬜ | ⬜ |
| FedAvg | 050IID | 25% | ⬜ | ⬜ | ⬜ |
| FedAvg | 050IID | 50% | ⬜ | ⬜ | ⬜ |
| FedAvg | 050IID | 75% | ⬜ | ⬜ | ⬜ |
| FedAvg | 000IID | 25% | ⬜ | ⬜ | ⬜ |
| FedAvg | 000IID | 50% | ⬜ | ⬜ | ⬜ |
| FedAvg | 000IID | 75% | ⬜ | ⬜ | ⬜ |
| FedYogi | 100IID | 25% | ⬜ | ⬜ | ⬜ |
| FedYogi | 100IID | 50% | ⬜ | ⬜ | ⬜ |
| FedYogi | 100IID | 75% | ⬜ | ⬜ | ⬜ |
| FedYogi | 050IID | 25% | ⬜ | ⬜ | ⬜ |
| FedYogi | 050IID | 50% | ⬜ | ⬜ | ⬜ |
| FedYogi | 050IID | 75% | ⬜ | ⬜ | ⬜ |
| FedYogi | 000IID | 25% | ⬜ | ⬜ | ⬜ |
| FedYogi | 000IID | 50% | ⬜ | ⬜ | ⬜ |
| FedYogi | 000IID | 75% | ⬜ | ⬜ | ⬜ |
| FedAdam | 100IID | 25% | ⬜ | ⬜ | ⬜ |
| FedAdam | 100IID | 50% | ⬜ | ⬜ | ⬜ |
| FedAdam | 100IID | 75% | ⬜ | ⬜ | ⬜ |
| FedAdam | 050IID | 25% | ⬜ | ⬜ | ⬜ |
| FedAdam | 050IID | 50% | ⬜ | ⬜ | ⬜ |
| FedAdam | 050IID | 75% | ⬜ | ⬜ | ⬜ |
| FedAdam | 000IID | 25% | ⬜ | ⬜ | ⬜ |
| FedAdam | 000IID | 50% | ⬜ | ⬜ | ⬜ |
| FedAdam | 000IID | 75% | ⬜ | ⬜ | ⬜ |
