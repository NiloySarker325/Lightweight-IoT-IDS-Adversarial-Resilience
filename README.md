Evaluating Adversarial Attack Resilience and Defensive Trade-Offs in Lightweight Machine Learning Intrusion Detection Systems for IoT Networks
Niloy Sarker Bappy
Department of Computer Information Science, Minnesota State University, Mankato, MN, USA
Abstract
The rapid expansion of Internet of Things (IoT) ecosystems has necessitated the deployment of lightweight Machine Learning (ML) Intrusion Detection Systems (IDS) capable of operating on resource-constrained edge gateways. While shallow neural networks and decision tree ensembles achieve near-perfect detection rates on clean benchmark traffic, their resilience against domain-constrained adversarial evasion remains a critical security bottleneck. In this paper, we evaluate the baseline performance, adversarial vulnerability, and defensive recovery of four lightweight classifiers—Decision Tree (DT), Random Forest (RF), XGBoost, and a 3-layer Multilayer Perceptron (MLP)—using a balanced 100,000-flow subset extracted from the CICIoT2023 dataset (Neto et al., 2023). Operating on an edge-representative hardware profile (Intel Core Ultra 5 235U, 16.0 GB RAM) without dedicated GPU acceleration, all baseline models achieve clean accuracies between 96.55% and 99.20% with sub-second batch inference latencies. However, under a white-box Fast Gradient Sign Method (FGSM) attack (ε = 0.1), the PyTorch MLP experiences an accuracy degradation of 10.42% to 11.44%. More severely, a black-box decision-based HopSkipJump attack (Chen et al., 2020) degrades the non-differentiable Random Forest classifier from 100.00% down to between 42.00% and 52.00% accuracy (a 48.00% to 58.00% drop) using only hard-label queries. Evaluating defensive countermeasures reveals a stark domain divergence: reactive Feature Squeezing (bit_depth = 4) (Xu et al., 2018) destroys fine-grained tabular flow signatures and further degrades MLP accuracy by 7.38%, whereas proactive Adversarial Training restores +11.58% in model resilience, achieving 98.18% accuracy under active FGSM attack with a retraining overhead of just 0.6969 seconds.

Index Terms — Adversarial Machine Learning, CICIoT2023, Edge Computing, Fast Gradient Sign Method (FGSM), Feature Squeezing, HopSkipJump, Internet of Things (IoT), Intrusion Detection System (IDS).
I. Introduction
The proliferation of interconnected Internet of Things (IoT) devices across residential, industrial, and healthcare infrastructures has vastly expanded the attack surface available to malicious actors (Neto et al., 2023). Modern IoT networks face a diverse spectrum of automated threats, ranging from high-volume Distributed Denial-of-Service (DDoS) and Denial-of-Service (DoS) floods (DDOS-TCP_FLOOD, DDOS-UDP_FLOOD, DOS-SYN_FLOOD) to Mirai botnet propagation (MIRAI-GREETH_FLOOD, MIRAI-UDPPLAIN), network reconnaissance (RECON-PORTSCAN, RECON-OSSCAN), Man-in-the-Middle ARP spoofing (MITM-ARPSPOOFING), and application-layer exploits such as SQL injection (SQLINJECTION) and cross-site scripting (XSS). Because traditional signature-based firewalls struggle to detect zero-day variations of these attacks, Machine Learning (ML) Intrusion Detection Systems (IDS) have become the primary line of defense for analyzing network flow telemetry.

However, deploying ML models within IoT architectures introduces strict computational constraints. Edge gateways and local fog nodes typically operate on low-power, general-purpose processors with limited memory and integrated graphics rather than high-wattage datacenter GPUs. Consequently, security practitioners rely on lightweight tabular classifiers—such as Decision Trees, Random Forests, XGBoost, and shallow Multilayer Perceptrons (MLPs)—that can classify tens of thousands of network flows in milliseconds.

Despite achieving high baseline accuracy on standard datasets, lightweight ML models are fundamentally vulnerable to adversarial machine learning (Nicolae et al., 2018). By introducing calculated, bounded perturbations into continuous network flow features (such as packet inter-arrival times, flow durations, or header lengths), an adversary can force an IDS to misclassify malicious traffic as benign without disrupting the underlying attack payload. Furthermore, many defensive mechanisms proposed in adversarial literature—such as input quantization and bit-depth squeezing (Xu et al., 2018)—were originally designed for computer vision tasks and have not been rigorously benchmarked for potential accuracy trade-offs on high-precision tabular network telemetry.

To address these challenges, this paper presents an empirical end-to-end evaluation of adversarial vulnerability and defensive recovery across lightweight IoT IDS architectures. The primary contributions of this work are as follows:

Balanced Pipeline Construction on CICIoT2023: We resolve the severe class imbalance inherent in raw IoT packet captures—where individual dataset partitions frequently contain zero benign records—by constructing a randomized, balanced 100,000-flow benchmark (50,000 BENIGN and 50,000 malicious samples across 39 scaled features) derived from the CICIoT2023 dataset (Neto et al., 2023).
Cross-Architecture Baseline and Latency Benchmarking: We evaluate four lightweight models (Decision Tree, Random Forest, XGBoost, and a 3-layer PyTorch MLP) on CPU-bound edge-representative hardware, demonstrating that while tree ensembles achieve peak clean accuracy (up to 99.20%), the shallow MLP achieves the fastest batch inference latency (0.0021 to 0.0047 seconds per 20,000 flows).
White-Box and Black-Box Evasion Analysis: We demonstrate that both differentiable neural networks and non-differentiable tree ensembles are vulnerable to evasion, showing a 10.42% to 11.44% accuracy drop on the MLP under white-box FGSM and a 48.00% to 58.00% accuracy collapse on the Random Forest under black-box HopSkipJump boundary probing (Chen et al., 2020).
Empirical Comparison of Reactive vs. Proactive Defenses: We prove that reactive Feature Squeezing (bit_depth = 4) is counterproductive for tabular IoT flows, inducing a negative resilience recovery (-7.38%), whereas proactive Adversarial Training completely neutralizes the FGSM threat (98.18% defended accuracy) with sub-second retraining overhead (0.6969 seconds).
II. Background and Related Work
A. Lightweight Machine Learning in IoT Intrusion Detection
Intrusion detection at the IoT edge requires balancing classification fidelity against hardware utilization. Deep learning architectures such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformers often incur excessive memory footprints and inference latencies when executed on CPU-only edge devices. In contrast, tree-based models (Decision Trees, Random Forests, and XGBoost) partition tabular feature spaces using discrete threshold splits, enabling rapid execution and high interpretability. Similarly, shallow Multilayer Perceptrons (MLPs) utilizing compact linear layers and Rectified Linear Unit (ReLU) activations offer vectorized matrix multiplication that executes in milliseconds even on standard integrated processors.
B. Benchmark Datasets for IoT Telemetry
Early intrusion detection studies relied heavily on legacy datasets such as KDD Cup 99, NSL-KDD, and UNSW-NB15, which lack modern IoT protocol structures and realistic botnet behaviors. To bridge this gap, Neto et al. (2023) introduced the CICIoT2023 dataset, generated from an authentic smart-home testbed comprising 105 real IoT devices (67 directly participating IP devices and 38 Zigbee/Z-Wave sensors connected via 5 hubs). The dataset captures 33 distinct attacks organized into seven overarching macro-categories: DDoS, DoS, Reconnaissance, Web-based exploits, Brute Force, Spoofing, and Mirai botnet operations.
C. Adversarial Evasion on Tabular Network Flows
Adversarial evasion attacks are categorized by the adversary's knowledge of the target classifier (Nicolae et al., 2018):

White-Box Attacks: The adversary possesses full access to the model architecture, parameters, and loss gradients. Gradient-based methods such as the Fast Gradient Sign Method (FGSM) perturb the input vector along the sign of the loss gradient to maximize classification error within a predefined L-infinity budget ε.
Black-Box Attacks: The adversary has no knowledge of the internal model structure and can only query the classifier as an oracle. Because decision tree ensembles lack continuous analytical gradients, white-box gradient attacks cannot be directly applied. Decision-based black-box algorithms such as HopSkipJump (Chen et al., 2020) overcome this barrier by initializing a sample in the target class and executing binary searches along the decision boundary, estimating gradient directions strictly from hard-label output flips.
D. Defensive Paradigms for Resource-Constrained IDS
Defending lightweight IDS models requires techniques that do not violate real-time processing budgets:

Input Transformation (Feature Squeezing): Originally developed to defend image classifiers (Xu et al., 2018), Feature Squeezing reduces the bit depth of continuous input features (e.g., mapping 32-bit floats to a 4-bit discrete scale) to coalesce adversarial perturbations with their unperturbed neighbors. While computationally inexpensive, its impact on tightly scaled tabular network statistics requires empirical validation.
Adversarial Training: Formulated as a min-max saddle-point optimization problem, adversarial training augments the training corpus with dynamically generated adversarial examples before re-optimizing the network weights. While computationally prohibitive for massive vision models, shallow tabular networks offer the potential for rapid adversarial retraining directly on edge hardware.
III. Experimental Methodology
A. Hardware Environment and Dataset Preprocessing
All experiments were conducted on a Dell Pro 16 PC16250 workstation equipped with an Intel Core Ultra 5 235U processor (2.00 GHz), 16.0 GB of 5600 MT/s RAM, and integrated Intel Graphics (128 MB). This hardware profile reflects the resource constraints of an IoT edge gateway or local fog node operating without dedicated high-wattage datacenter accelerators.

To evaluate intrusion detection resilience under realistic network conditions, data was extracted from the CICIoT2023 dataset (Neto et al., 2023), which comprises flow-based tabular features across BENIGN traffic and 33 distinct malicious attack categories (including DDoS floods, Mirai botnet traffic, reconnaissance scans, and spoofing attacks). Because raw IoT traffic captures exhibit severe class imbalance—frequently containing zero benign records across individual CSV partitions—an automated sampling pipeline was implemented to scan sequential dataset partitions (Merged01.csv through Merged03.csv) until an exact 50/50 balanced subset of 100,000 records (50,000 benign and 50,000 malicious flows) across 40 columns was constructed. Multi-class attack strings were mapped into a binary classification target y in {0, 1}, where 0 denotes BENIGN traffic and 1 denotes an active attack.

After replacing infinite values and dropping corrupted records, the resulting dataset contained 39 continuous and discrete network flow features. To prevent high-magnitude features from dominating gradient updates and to establish strict bounded domain constraints for adversarial generation, all features were normalized into the range [0.0, 1.0] using Min-Max scaling:

x' = (x − min(x)) / (max(x) − min(x))

The cleaned, normalized dataset was partitioned via an 80/20 split into 79,995 training samples and 19,999 testing samples.
B. Lightweight Baseline Architectures
Four lightweight machine learning classifiers suitable for low-latency IoT deployment were trained and benchmarked:

Decision Tree (DT): Configured as a fast, rule-based baseline to measure single-tree decision boundary stability.
Random Forest (RF): An ensemble of decision trees trained on stripped NumPy feature arrays (.values) to eliminate metadata overhead and evaluate bagging resilience against black-box boundary probing.
XGBoost: A gradient-boosted decision tree ensemble optimized for high tabular classification accuracy.
Lightweight Multilayer Perceptron (MLP): A 3-layer feed-forward neural network implemented in PyTorch utilizing 32-bit floating-point weights, ReLU activations, a sigmoid output layer, Binary Cross-Entropy (BCE) loss, and the Adam optimizer (learning rate = 0.001).
C. Threat Model and Adversarial Attack Formulation
We evaluate two distinct threat models using the Adversarial Robustness Toolbox (ART) (Nicolae et al., 2018):

1) White-Box Gradient Attack (FGSM on MLP): Assuming the adversary has full knowledge of the neural network architecture and weights, we apply the Fast Gradient Sign Method (FGSM). FGSM computes the gradient of the loss function J(θ, x, y) with respect to the input feature vector x and applies a single-step perturbation of magnitude ε = 0.1, clipped to the valid normalized feature space [0.0, 1.0] across a 5,000-sample test subset:

x_adv = clip_[0.0, 1.0](x + ε · sign(∇_x J(θ, x, y)))

2) Black-Box Decision-Based Attack (HopSkipJump on Random Forest): Because tree-based ensembles are non-differentiable and lack continuous gradients, gradient-based evasion cannot be directly applied. Instead, we execute the black-box HopSkipJump algorithm (Chen et al., 2020) against the Random Forest classifier across a 100-sample test subset. The attacker observes only the final hard-label predictions (0 or 1) and iteratively estimates the gradient direction at the decision boundary via binary search and Monte Carlo boundary sampling (max_iter = 10, max_eval = 1000, init_eval = 10).
D. Defensive Countermeasures
Two distinct defensive paradigms were evaluated on the MLP architecture:

Reactive Input Transformation (Feature Squeezing): Input features of the 5,000 adversarial test flows were compressed to a reduced precision of bit_depth = 4 inside the [0.0, 1.0] interval prior to inference and explicitly cast back to 32-bit floating-point tensors (np.float32) to match PyTorch layer weights (Xu et al., 2018).
Proactive Model Hardening (Adversarial Training): FGSM perturbations (ε = 0.1) were generated across a 10,000-sample subset of the training data and concatenated with the clean records to form a 20,000-sample augmented training set. A new instance of the PyTorch MLP was trained for 50 epochs to solve the min-max robust optimization objective:

min_θ E_(x,y) [ max_(||δ||_∞ ≤ ε) J(θ, x + δ, y) ]
|---|---|---|
IV. Experimental Results and Analysis
A. Clean Baseline Performance and Latency
All four lightweight classifiers achieved strong baseline detection accuracy above 96.5% on the unperturbed 19,999-sample test set while maintaining sub-second batch inference times.
Table I: Clean Baseline Classification Performance and Batch Inference Latency (20k Test Samples)
Model Architecture
Clean Accuracy (Range Across Runs)
Clean F1-Score (Range Across Runs)
Batch Inference Latency (20k Samples)
Per-Sample Latency (Microseconds)
Decision Tree (DT)
98.51%
0.9850
0.0109 – 0.0196 s
0.55 – 0.98 µs
Random Forest (RF)
99.20%
0.9919
0.0394 – 0.1075 s
1.97 – 5.38 µs
XGBoost
99.08%
0.9906
0.0124 – 0.0408 s
0.62 – 2.04 µs
PyTorch MLP (3-Layer)
96.55% – 98.14%
0.9658 – 0.9810
0.0021 – 0.0047 s
0.11 – 0.24 µs



Figure 2: Multi-metric baseline comparison of Accuracy, F1-Score, and Batch Inference Latency (ms) across Decision Tree, Random Forest, XGBoost, and 3-Layer PyTorch MLP architectures on 20,000 test flows.

As shown in Table I, ensemble tree architectures achieved the highest initial classification accuracy, led by Random Forest at 99.20% and XGBoost at 99.08%. However, the 3-layer PyTorch MLP demonstrated the lowest inference latency, processing nearly 20,000 network flows in just 0.0021 to 0.0047 seconds (0.11 to 0.24 microseconds per flow), making it exceptionally well-suited for high-throughput IoT edge interfaces.
B. Vulnerability to White-Box and Black-Box Evasion
Despite near-perfect baseline metrics on clean traffic, both neural and tree-based models exhibited severe degradation when subjected to white-box and black-box adversarial evasion attacks.

Figure 1: Empirical comparison of white-box FGSM degradation and Adversarial Training recovery on the 3-layer PyTorch MLP (left) versus black-box HopSkipJump degradation on the Random Forest classifier (right).
Table II: Adversarial Evasion Vulnerability and Defensive Recovery Benchmarks
Target Model & Evaluation Scenario
Clean Subset Accuracy
Attacked / Defended Accuracy
Net Accuracy Change
Execution / Overhead Time
MLP under White-Box FGSM (ε = 0.1, 5k subset, Run 1)
97.86%
86.42%
-11.44%
Instantaneous
MLP under White-Box FGSM (ε = 0.1, 5k subset, Final Run)
97.02%
86.60%
-10.42%
Instantaneous
RF under Black-Box HopSkipJump (Single-Core, 100 subset)
100.00%
50.00%
-50.00%
244.0 s (2.45 s/it)
RF under Black-Box HopSkipJump (Multi-Core, 100 subset, Run 1)
100.00%
42.00%
-58.00%
~100.0 s
RF under Black-Box HopSkipJump (Multi-Core, 100 subset, Run 2)
100.00%
52.00%
-48.00%
100.0 s (1.00 s/it)
MLP Defense: Feature Squeezing (bit_depth = 4, 5k subset)
96.72% (93.86% FGSM)
86.48%
-7.38% (vs. FGSM)
0.0059 s
MLP Defense: Adversarial Training (50 epochs, 20k aug.)
97.82% (86.60% FGSM)
98.18%
+11.58% (Recovered)
0.6969 s

In the white-box evaluation, applying an FGSM perturbation with ε = 0.1 reduced the PyTorch MLP&apos;s detection accuracy from 97.02% down to 86.60% (a 10.42% drop in the final multi-threaded run, and an 11.44% drop from 97.86% to 86.42% in initial testing), as illustrated in Figure 1 (left) and Table II.

Figure 3: Black-box HopSkipJump attack effectiveness and runtime optimization comparing Single-Core execution (244.0 s, 50.0% accuracy) versus Multi-Core stripped-array execution (100.0 s, 42.0%–52.0% accuracy).

More critically, the black-box HopSkipJump attack demonstrated that non-differentiable tree ensembles do not provide inherent security through obscurity (Chen et al., 2020). Without access to internal tree splits or training data, HopSkipJump degraded the Random Forest&apos;s accuracy from 100.00% down to 50.00% under single-core execution and between 42.00% and 52.00% (a 48.00% to 58.00% accuracy drop) across multi-threaded runs, as depicted in Figure 1 (right) and Figure 3. Furthermore, optimizing the pipeline to strip DataFrame column metadata (.values) and utilize multi-core parallelization (n_jobs = -1) reduced the black-box attack generation time for 100 samples from 4 minutes and 4 seconds (2.45 s/iteration) to 1 minute and 40 seconds (1.00 s/iteration)—a 2.44x speedup.
C. Efficacy and Trade-Offs of Defensive Mechanisms
A key finding of this study is the stark divergence between reactive input quantization and proactive adversarial retraining when applied to tabular IoT flow metrics:

Figure 4: Comparative defense divergence on the PyTorch MLP showing negative resilience recovery under reactive Feature Squeezing (-7.38%) versus full resilience restoration under proactive Adversarial Training (+11.58%).
Failure of Feature Squeezing on Tabular Flows: Although Feature Squeezing (Xu et al., 2018) introduced a negligible computational overhead of 0.0059 seconds across 5,000 samples, it yielded a negative resilience recovery of -0.0738 (-7.38%), as shown in Figure 4. Specifically, in the evaluated trial, squeezing the 32-bit floating-point features to 4-bit precision caused accuracy on the FGSM-perturbed flows to fall further from 93.86% down to 86.48%. While bit-depth reduction is effective in computer vision where adjacent pixel intensities are spatially redundant, continuous network telemetry features (such as packet inter-arrival times and header lengths) rely on fine-grained numeric precision; coarse quantization destroys benign traffic signatures alongside adversarial perturbations.
Success of Adversarial Training: Conversely, retraining the lightweight MLP on an augmented dataset of clean and FGSM-perturbed samples completely neutralized the white-box threat. The adversarially trained MLP maintained a 97.82% accuracy on clean traffic while achieving 98.18% accuracy under active FGSM attack—recovering +11.58% in accuracy over the unhardened model&apos;s 86.60% attacked state. Crucially, the entire 50-epoch retraining process required only 0.6969 seconds of computation overhead, confirming that proactive adversarial training is both highly effective and computationally lightweight enough for periodic model updates on edge IoT hardware.

Figure 5: Logarithmic-scale computational timing comparison across baseline batch inference (20k flows), Feature Squeezing preprocessing (5k flows), Adversarial Training (50 epochs, 20k flows), and Black-Box HopSkipJump boundary querying (100 flows).
V. Conclusion and Future Work
This study investigated the resilience of lightweight machine learning intrusion detection systems against white-box and black-box adversarial evasion using a balanced 100,000-flow benchmark derived from the CICIoT2023 dataset (Neto et al., 2023). While baseline evaluations confirmed that Decision Trees, Random Forests, XGBoost, and a 3-layer PyTorch MLP achieve exceptional clean detection accuracies (96.55% to 99.20%) and sub-second inference speeds on resource-constrained hardware, adversarial testing exposed critical vulnerabilities across both model families. A white-box FGSM attack (ε = 0.1) degraded the MLP&apos;s accuracy by up to 11.44%, while a black-box HopSkipJump attack reduced the Random Forest&apos;s detection rate by up to 58.00% without requiring internal model access.
Furthermore, our evaluation of lightweight defensive mechanisms revealed that countermeasures cannot be blindly transferred from computer vision to network security. Reactive Feature Squeezing (bit_depth = 4), despite executing in just 0.0059 seconds, degraded the MLP&apos;s accuracy under attack by an additional 7.38% due to the loss of fine-grained numerical precision across scaled tabular flow features. Conversely, proactive Adversarial Training proved highly effective and computationally viable, recovering +11.58% in model resilience to achieve a 98.18% detection rate against active FGSM payloads while requiring only 0.6969 seconds of retraining overhead.
Future work will extend this framework in three directions:
Multi-Class Adversarial Evaluation: Expanding the binary classification pipeline to evaluate targeted evasion across all 34 individual traffic classes in the CICIoT2023 dataset.
Hardening Tree Ensembles: Investigating adversarial boosting and robust tree-splitting criteria to defend Random Forest and XGBoost architectures against black-box HopSkipJump boundary probing.
Real-Time Edge Deployment: Integrating the adversarially trained PyTorch MLP into a live packet-capture daemon to measure end-to-end throughput and power consumption on physical IoT gateway nodes.
References
Apruzzese, G., Colajanni, M., Ferretti, L., Guido, A., & Marchetti, M. (2018). On the effectiveness of machine and deep learning for cyber security. In 2018 10th International Conference on Cyber Conflict (CyCon) (pp. 371–390). IEEE.
Apruzzese, G., Andreolini, M., Ferretti, L., Marchetti, M., & Colajanni, M. (2022). Modeling realistic adversarial attacks against network intrusion detection systems. Digital Threats: Research and Practice, 3(3), 1–19.
Biggio, B., Corona, I., Maiorca, D., Nelson, B., Šrndić, N., Laskov, P., Giacinto, G., & Roli, F. (2013). Evasion attacks against machine learning at test time. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD) (pp. 387–402). Springer.
Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.
Brendel, W., Rauber, J., & Bethge, M. (2018). Decision-based adversarial attacks: Reliable attacks against black-box machine learning models. In International Conference on Learning Representations (ICLR).
Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy (SP) (pp. 39–57). IEEE.
Chen, J., Jordan, M. I., & Wainwright, M. J. (2020). HopSkipJumpAttack: A query-efficient decision-based attack. In 2020 IEEE Symposium on Security and Privacy (SP) (pp. 1277–1294). IEEE.
Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). ACM.
Corona, I., Giacinto, G., & Roli, F. (2013). Adversarial attacks against intrusion detection systems: Taxonomy, solutions and open issues. Information Sciences, 239, 201–225.
Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. In International Conference on Learning Representations (ICLR).
He, K., Kim, D. D., & Asghar, M. R. (2023). Adversarial machine learning for network intrusion detection systems: A comprehensive survey. IEEE Communications Surveys & Tutorials, 25(1), 538–566.
Ibitoye, O., Shafiq, O., & Matrawy, A. (2019). Analyzing adversarial attacks against deep learning for intrusion detection in IoT networks. In 2019 IEEE Global Communications Conference (GLOBECOM) (pp. 1–6). IEEE.
Kolias, C., Kambourakis, G., Stavrou, A., & Voas, J. (2017). DDoS in the IoT: Mirai and other botnets. Computer, 50(7), 80–84.
Kantchelian, A., Tygar, J. D., & Joseph, A. (2016). Evasion and hardening of tree ensemble classifiers. In International Conference on Machine Learning (ICML) (pp. 2387–2396). PMLR.
Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations (ICLR).
Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems. In 2015 Military Communications and Information Systems Conference (MilCIS) (pp. 1–6). IEEE.
Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. A. (2023). CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment. Sensors, 23(13), 5941.
Nicolae, M.-I., Sinn, M., Tran, M. N., Buesser, B., Rawat, A., Wistuba, M., Zantedeschi, V., Baracaldo, N., Chen, B., Ludwig, H., Molloy, I. M., & Edwards, B. (2018). Adversarial Robustness Toolbox v1.0.0. arXiv preprint arXiv:1807.01069.
Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z. B., & Swami, A. (2016). The limitations of deep learning in adversarial settings. In 2016 IEEE European Symposium on Security and Privacy (EuroS&P) (pp. 372–387). IEEE.
Papernot, N., McDaniel, P., Goodfellow, I., Jha, S., Celik, Z. B., & Swami, A. (2017). Practical black-box attacks against machine learning. In Proceedings of the 2017 ACM on Asia Conference on Computer and Communications Security (ASIACCS) (pp. 506–519). ACM.
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems (NeurIPS) (Vol. 32).
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
Quinlan, J. R. (1986). Induction of decision trees. Machine Learning, 1(1), 81–106.
Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). Intriguing properties of neural networks. In International Conference on Learning Representations (ICLR).
Xu, W., Evans, D., & Qi, Y. (2018). Feature squeezing: Detecting adversarial examples in deep neural networks. In Proceedings of the 2018 Network and Distributed System Security Symposium (NDSS).
