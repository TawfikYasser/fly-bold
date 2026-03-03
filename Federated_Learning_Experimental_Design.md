
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

---

### Flower Experiments

| EXP_ID   | Strategy | Dataset | Fail % | Status |
|----------|----------|---------|--------|--------|
| [11213141](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213141) | FedAvg  | 100IID  | 25%    | ✅ |
| [11213142](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213142) | FedAvg  | 100IID  | 50%    | ✅ |
| [11213143](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213143) | FedAvg  | 100IID  | 75%    | ✅ |
| [11213241](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213241) | FedAvg  | 050IID  | 25%    | ✅ |
| [11213242](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213242) | FedAvg  | 050IID  | 50%    | ✅ |
| [11213243](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213243) | FedAvg  | 050IID  | 75%    | ✅ |
| [11213341](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213341) | FedAvg  | 000IID  | 25%    | ✅ |
| [11213342](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213342) | FedAvg  | 000IID  | 50%    | ✅ |
| [11213343](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213343) | FedAvg  | 000IID  | 75%    | 🟡 |
| 11223141 | FedYogi | 100IID  | 25%    | ⬜ |
| 11223142 | FedYogi | 100IID  | 50%    | ⬜ |
| 11223143 | FedYogi | 100IID  | 75%    | ⬜ |
| 11223241 | FedYogi | 050IID  | 25%    | ⬜ |
| 11223242 | FedYogi | 050IID  | 50%    | ⬜ |
| 11223243 | FedYogi | 050IID  | 75%    | ⬜ |
| 11223341 | FedYogi | 000IID  | 25%    | ⬜ |
| 11223342 | FedYogi | 000IID  | 50%    | ⬜ |
| 11223343 | FedYogi | 000IID  | 75%    | ⬜ |
| 11233141 | FedAdam | 100IID  | 25%    | ⬜ |
| 11233142 | FedAdam | 100IID  | 50%    | ⬜ |
| 11233143 | FedAdam | 100IID  | 75%    | ⬜ |
| 11233241 | FedAdam | 050IID  | 25%    | ⬜ |
| 11233242 | FedAdam | 050IID  | 50%    | ⬜ |
| 11233243 | FedAdam | 050IID  | 75%    | ⬜ |
| 11233341 | FedAdam | 000IID  | 25%    | ⬜ |
| 11233342 | FedAdam | 000IID  | 50%    | ⬜ |
| 11233343 | FedAdam | 000IID  | 75%    | ⬜ |

---
### FEDn Experiments

| EXP_ID   | Strategy | Dataset | Fail % | Status |
|----------|----------|---------|--------|--------|
| 12213141 | FedAvg  | 100IID  | 25%    | ⬜ |
| 12213142 | FedAvg  | 100IID  | 50%    | ⬜ |
| 12213143 | FedAvg  | 100IID  | 75%    | ⬜ |
| 12213241 | FedAvg  | 050IID  | 25%    | ⬜ |
| 12213242 | FedAvg  | 050IID  | 50%    | ⬜ |
| 12213243 | FedAvg  | 050IID  | 75%    | ⬜ |
| 12213341 | FedAvg  | 000IID  | 25%    | ⬜ |
| 12213342 | FedAvg  | 000IID  | 50%    | ⬜ |
| 12213343 | FedAvg  | 000IID  | 75%    | ⬜ |
| 12223141 | FedYogi | 100IID  | 25%    | ⬜ |
| 12223142 | FedYogi | 100IID  | 50%    | ⬜ |
| 12223143 | FedYogi | 100IID  | 75%    | ⬜ |
| 12223241 | FedYogi | 050IID  | 25%    | ⬜ |
| 12223242 | FedYogi | 050IID  | 50%    | ⬜ |
| 12223243 | FedYogi | 050IID  | 75%    | ⬜ |
| 12223341 | FedYogi | 000IID  | 25%    | ⬜ |
| 12223342 | FedYogi | 000IID  | 50%    | ⬜ |
| 12223343 | FedYogi | 000IID  | 75%    | ⬜ |
| 12233141 | FedAdam | 100IID  | 25%    | ⬜ |
| 12233142 | FedAdam | 100IID  | 50%    | ⬜ |
| 12233143 | FedAdam | 100IID  | 75%    | ⬜ |
| 12233241 | FedAdam | 050IID  | 25%    | ⬜ |
| 12233242 | FedAdam | 050IID  | 50%    | ⬜ |
| 12233243 | FedAdam | 050IID  | 75%    | ⬜ |
| 12233341 | FedAdam | 000IID  | 25%    | ⬜ |
| 12233342 | FedAdam | 000IID  | 50%    | ⬜ |
| 12233343 | FedAdam | 000IID  | 75%    | ⬜ |

---
### FedML Experiments

| EXP_ID   | Strategy | Dataset | Fail % | Status |
|----------|----------|---------|--------|--------|
| 13213141 | FedAvg  | 100IID  | 25%    | ⬜ |
| 13213142 | FedAvg  | 100IID  | 50%    | ⬜ |
| 13213143 | FedAvg  | 100IID  | 75%    | ⬜ |
| 13213241 | FedAvg  | 050IID  | 25%    | ⬜ |
| 13213242 | FedAvg  | 050IID  | 50%    | ⬜ |
| 13213243 | FedAvg  | 050IID  | 75%    | ⬜ |
| 13213341 | FedAvg  | 000IID  | 25%    | ⬜ |
| 13213342 | FedAvg  | 000IID  | 50%    | ⬜ |
| 13213343 | FedAvg  | 000IID  | 75%    | ⬜ |
| 13223141 | FedYogi | 100IID  | 25%    | ⬜ |
| 13223142 | FedYogi | 100IID  | 50%    | ⬜ |
| 13223143 | FedYogi | 100IID  | 75%    | ⬜ |
| 13223241 | FedYogi | 050IID  | 25%    | ⬜ |
| 13223242 | FedYogi | 050IID  | 50%    | ⬜ |
| 13223243 | FedYogi | 050IID  | 75%    | ⬜ |
| 13223341 | FedYogi | 000IID  | 25%    | ⬜ |
| 13223342 | FedYogi | 000IID  | 50%    | ⬜ |
| 13223343 | FedYogi | 000IID  | 75%    | ⬜ |
| 13233141 | FedAdam | 100IID  | 25%    | ⬜ |
| 13233142 | FedAdam | 100IID  | 50%    | ⬜ |
| 13233143 | FedAdam | 100IID  | 75%    | ⬜ |
| 13233241 | FedAdam | 050IID  | 25%    | ⬜ |
| 13233242 | FedAdam | 050IID  | 50%    | ⬜ |
| 13233243 | FedAdam | 050IID  | 75%    | ⬜ |
| 13233341 | FedAdam | 000IID  | 25%    | ⬜ |
| 13233342 | FedAdam | 000IID  | 50%    | ⬜ |
| 13233343 | FedAdam | 000IID  | 75%    | ⬜ |