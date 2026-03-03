
# 🧪 Federated Learning Experimental Design

## 0. Experimental Setup Constrains

Our setup is deployed on GCP VMs. We use one VM for the server and five VMs for the clients. The server VM is configured as e2-standard-8 (8 vCPUs, 32 GB RAM). Each client VM is configured as e2-standard-16 (16 vCPUs, 64 GB RAM). We are using CPU only.

Based on this setup, the average experiment duration is approximately 12 hours. **Due to time and cost constraints**, we fixed the training parameters to match our available resources. Specifically, we train for 15 rounds with 3 local epochs per round, using datasets of up to 5,000 images.

We constructed three datasets:
- dataset_100: Partitioned among 10 clients. Each client is assigned a very large alpha value (1e6), ensuring that the Dirichlet partitioner distributes an equal number of samples from each class to all clients. This simulates a fully IID scenario.
- dataset_050: Contains 10 clients. Five clients are assigned alpha = 1e6 to simulate IID behavior (as in dataset_100). The remaining five clients are assigned randomly generated alpha values in the range [0.5, 1.5] during the partitioning phase, simulating a partially IID setting (50% IID and 50% non-IID).
- dataset_000: Partitioned among 10 clients, each assigned a randomly generated alpha value in the range [0.5, 1.5], simulating a fully non-IID scenario.

====> Concluding that, given all constraints, the current experiments represent the most suitable configuration for our environment.

**Dr. Feras notes:**
1. Centralized baseline
2. Server rounds between 50 and 100
3. Metrics must include: 
    a. communication costs
    b. Resource utilization: CPU/Memory
    c. Runtime per round
4. Heterogeneity of IID: 0.1, 0.01, 
5. Clients sampling <-> Client failure
6. Clients 20,50,100
7. Estimated cost per accuracy gain
NOTE: Different seeds for each configuration and report the mean and STDV

---

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
- EXP_ID: redirects to the directory which contains the logs and final model of the experiment.

- Results: redirects to the directory which contains the summary and plots of the experiment.

- Strategy, Dataset, & Fail % are Dim_2, Dim_3, & Dim_4. Below are 3 tables for Dim_1.

---

### Flower Experiments

| EXP_ID   | Strategy | Dataset | Fail % | Status | Results |
|----------|----------|---------|--------|--------|-------------|
| [11213141](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213141) | FedAvg  | 100IID  | 25%    | ✅ | [11213141](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11213141) |
| [11213142](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213142) | FedAvg  | 100IID  | 50%    | ✅ | [11213142](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11213142) |
| [11213143](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213143) | FedAvg  | 100IID  | 75%    | ✅ | [11213143](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11213143) |
| [11213241](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213241) | FedAvg  | 050IID  | 25%    | ✅ | [11213241](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11213241) |
| [11213242](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213242) | FedAvg  | 050IID  | 50%    | ✅ | [11213242](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11213242) |
| [11213243](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213243) | FedAvg  | 050IID  | 75%    | ✅ | [11213243](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11213243) |
| [11213341](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213341) | FedAvg  | 000IID  | 25%    | ✅ | [11213341](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11213341) |
| [11213342](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213342) | FedAvg  | 000IID  | 50%    | ✅ | [11213342](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11213342) |
| [11213343](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11213343) | FedAvg  | 000IID  | 75%    | ✅ | [11213343](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11213343) |
| [11223141](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11223141) | FedYogi | 100IID  | 25%    | ⬜ | [11223141](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11223141) |
| [11223142](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11223142) | FedYogi | 100IID  | 50%    | ⬜ | [11223142](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11223142) |
| [11223143](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11223143) | FedYogi | 100IID  | 75%    | ⬜ | [11223143](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11223143) |
| [11223241](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11223241) | FedYogi | 050IID  | 25%    | ⬜ | [11223241](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11223241) |
| [11223242](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11223242) | FedYogi | 050IID  | 50%    | ⬜ | [11223242](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11223242) |
| [11223243](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11223243) | FedYogi | 050IID  | 75%    | ⬜ | [11223243](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11223243) |
| [11223341](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11223341) | FedYogi | 000IID  | 25%    | ⬜ | [11223341](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11223341) |
| [11223342](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11223342) | FedYogi | 000IID  | 50%    | ⬜ | [11223342](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11223342) |
| [11223343](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11223343) | FedYogi | 000IID  | 75%    | ⬜ | [11223343](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11223343) |
| [11233141](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11233141) | FedAdam | 100IID  | 25%    | ⬜ | [11233141](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11233141) |
| [11233142](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11233142) | FedAdam | 100IID  | 50%    | ⬜ | [11233142](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11233142) |
| [11233143](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11233143) | FedAdam | 100IID  | 75%    | ⬜ | [11233143](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11233143) |
| [11233241](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11233241) | FedAdam | 050IID  | 25%    | ⬜ | [11233241](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11233241) |
| [11233242](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11233242) | FedAdam | 050IID  | 50%    | ⬜ | [11233242](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11233242) |
| [11233243](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11233243) | FedAdam | 050IID  | 75%    | ⬜ | [11233243](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11233243) |
| [11233341](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11233341) | FedAdam | 000IID  | 25%    | ⬜ | [11233341](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11233341) |
| [11233342](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11233342) | FedAdam | 000IID  | 50%    | ⬜ | [11233342](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11233342) |
| [11233343](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/11233343) | FedAdam | 000IID  | 75%    | ⬜ | [11233343](https://github.com/TawfikYasser/fly-bold/tree/main/flower/experiments_outputs/analysis_exp_11233343) |}

---
### FEDn Experiments

| EXP_ID   | Strategy | Dataset | Fail % | Status | EXP_Results |
|----------|----------|---------|--------|--------|-------------|
| [12213141](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12213141) | FedAvg  | 100IID  | 25%    | ⬜ | [12213141](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12213141) |
| [12213142](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12213142) | FedAvg  | 100IID  | 50%    | ⬜ | [12213142](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12213142) |
| [12213143](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12213143) | FedAvg  | 100IID  | 75%    | ⬜ | [12213143](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12213143) |
| [12213241](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12213241) | FedAvg  | 050IID  | 25%    | ⬜ | [12213241](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12213241) |
| [12213242](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12213242) | FedAvg  | 050IID  | 50%    | ⬜ | [12213242](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12213242) |
| [12213243](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12213243) | FedAvg  | 050IID  | 75%    | ⬜ | [12213243](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12213243) |
| [12213341](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12213341) | FedAvg  | 000IID  | 25%    | ⬜ | [12213341](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12213341) |
| [12213342](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12213342) | FedAvg  | 000IID  | 50%    | ⬜ | [12213342](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12213342) |
| [12213343](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12213343) | FedAvg  | 000IID  | 75%    | ⬜ | [12213343](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12213343) |
| [12223141](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12223141) | FedYogi | 100IID  | 25%    | ⬜ | [12223141](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12223141) |
| [12223142](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12223142) | FedYogi | 100IID  | 50%    | ⬜ | [12223142](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12223142) |
| [12223143](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12223143) | FedYogi | 100IID  | 75%    | ⬜ | [12223143](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12223143) |
| [12223241](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12223241) | FedYogi | 050IID  | 25%    | ⬜ | [12223241](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12223241) |
| [12223242](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12223242) | FedYogi | 050IID  | 50%    | ⬜ | [12223242](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12223242) |
| [12223243](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12223243) | FedYogi | 050IID  | 75%    | ⬜ | [12223243](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12223243) |
| [12223341](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12223341) | FedYogi | 000IID  | 25%    | ⬜ | [12223341](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12223341) |
| [12223342](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12223342) | FedYogi | 000IID  | 50%    | ⬜ | [12223342](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12223342) |
| [12223343](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12223343) | FedYogi | 000IID  | 75%    | ⬜ | [12223343](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12223343) |
| [12233141](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12233141) | FedAdam | 100IID  | 25%    | ⬜ | [12233141](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12233141) |
| [12233142](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12233142) | FedAdam | 100IID  | 50%    | ⬜ | [12233142](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12233142) |
| [12233143](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12233143) | FedAdam | 100IID  | 75%    | ⬜ | [12233143](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12233143) |
| [12233241](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12233241) | FedAdam | 050IID  | 25%    | ⬜ | [12233241](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12233241) |
| [12233242](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12233242) | FedAdam | 050IID  | 50%    | ⬜ | [12233242](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12233242) |
| [12233243](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12233243) | FedAdam | 050IID  | 75%    | ⬜ | [12233243](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12233243) |
| [12233341](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12233341) | FedAdam | 000IID  | 25%    | ⬜ | [12233341](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12233341) |
| [12233342](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12233342) | FedAdam | 000IID  | 50%    | ⬜ | [12233342](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12233342) |
| [12233343](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/12233343) | FedAdam | 000IID  | 75%    | ⬜ | [12233343](https://github.com/TawfikYasser/fly-bold/tree/main/fedn/experiments_outputs/analysis_exp_12233343) |

---
### FedML Experiments

| EXP_ID   | Strategy | Dataset | Fail % | Status | EXP_Results |
|----------|----------|---------|--------|--------|-------------|
| [13213141](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13213141) | FedAvg  | 100IID  | 25%    | ⬜ | [13213141](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13213141) |
| [13213142](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13213142) | FedAvg  | 100IID  | 50%    | ⬜ | [13213142](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13213142) |
| [13213143](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13213143) | FedAvg  | 100IID  | 75%    | ⬜ | [13213143](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13213143) |
| [13213241](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13213241) | FedAvg  | 050IID  | 25%    | ⬜ | [13213241](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13213241) |
| [13213242](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13213242) | FedAvg  | 050IID  | 50%    | ⬜ | [13213242](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13213242) |
| [13213243](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13213243) | FedAvg  | 050IID  | 75%    | ⬜ | [13213243](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13213243) |
| [13213341](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13213341) | FedAvg  | 000IID  | 25%    | ⬜ | [13213341](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13213341) |
| [13213342](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13213342) | FedAvg  | 000IID  | 50%    | ⬜ | [13213342](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13213342) |
| [13213343](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13213343) | FedAvg  | 000IID  | 75%    | ⬜ | [13213343](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13213343) |
| [13223141](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13223141) | FedYogi | 100IID  | 25%    | ⬜ | [13223141](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13223141) |
| [13223142](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13223142) | FedYogi | 100IID  | 50%    | ⬜ | [13223142](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13223142) |
| [13223143](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13223143) | FedYogi | 100IID  | 75%    | ⬜ | [13223143](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13223143) |
| [13223241](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13223241) | FedYogi | 050IID  | 25%    | ⬜ | [13223241](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13223241) |
| [13223242](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13223242) | FedYogi | 050IID  | 50%    | ⬜ | [13223242](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13223242) |
| [13223243](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13223243) | FedYogi | 050IID  | 75%    | ⬜ | [13223243](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13223243) |
| [13223341](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13223341) | FedYogi | 000IID  | 25%    | ⬜ | [13223341](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13223341) |
| [13223342](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13223342) | FedYogi | 000IID  | 50%    | ⬜ | [13223342](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13223342) |
| [13223343](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13223343) | FedYogi | 000IID  | 75%    | ⬜ | [13223343](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13223343) |
| [13233141](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13233141) | FedAdam | 100IID  | 25%    | ⬜ | [13233141](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13233141) |
| [13233142](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13233142) | FedAdam | 100IID  | 50%    | ⬜ | [13233142](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13233142) |
| [13233143](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13233143) | FedAdam | 100IID  | 75%    | ⬜ | [13233143](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13233143) |
| [13233241](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13233241) | FedAdam | 050IID  | 25%    | ⬜ | [13233241](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13233241) |
| [13233242](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13233242) | FedAdam | 050IID  | 50%    | ⬜ | [13233242](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13233242) |
| [13233243](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13233243) | FedAdam | 050IID  | 75%    | ⬜ | [13233243](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13233243) |
| [13233341](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13233341) | FedAdam | 000IID  | 25%    | ⬜ | [13233341](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13233341) |
| [13233342](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13233342) | FedAdam | 000IID  | 50%    | ⬜ | [13233342](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13233342) |
| [13233343](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/13233343) | FedAdam | 000IID  | 75%    | ⬜ | [13233343](https://github.com/TawfikYasser/fly-bold/tree/main/fedml/experiments_outputs/analysis_exp_13233343) |