
The current robust experiments map:

| EXP_ID⇘ | Optuna ONLY | BOHB |
|----------|----------|----------|
| Dataset_100 (Fully IID)   | 103101004   | 3010101001   |
| Dataset_000 (Fully non-IID)   | 103100001   | 3010101002   |

=> BHOB pruner configs: min = 1, max = hop_rounds, and rf = 3

---

### FILES

* [103101004 Results Analysis](https://github.com/TawfikYasser/fly-bold/blob/main/results_103101004.html)
* [103100001 Results Analysis](https://github.com/TawfikYasser/fly-bold/blob/main/results_103100001.html)
* [103101004 Run Logs](https://github.com/TawfikYasser/fly-bold/blob/main/103101004.txt)
* [103100001 Run Logs](https://github.com/TawfikYasser/fly-bold/blob/main/103100001.txt)
* [103101004 HPO DB File](https://github.com/TawfikYasser/fly-bold/blob/main/EXP_YOLOv5_s_detection_103101004_hpo.db)
* [103100001 HPO DB File](https://github.com/TawfikYasser/fly-bold/blob/main/EXP_YOLOv5_s_detection_103100001_hpo.db)

For more details on the pruner configs, see [BOHB documentation](https://github.com/automl/HpBandSter).


