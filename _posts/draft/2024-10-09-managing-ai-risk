---
layout: post
title:  "Managing AI Risks"
date:   2024-10-09 07:01:55 +0530
categories: journals
---
# Abstract

Artificial Intelligence (AI) is progressing rapidly, and companies are shifting their focus to developing generalist AI systems that can autonomously act and pursue goals. Increases in capabilities and autonomy may soon massively amplify AI’s impact, with risks that include large-scale social harms, malicious uses, and an irreversible loss of human control over autonomous AI systems. Although researchers have warned of extreme risks from AI1, there is a lack of consensus about how exactly such risk arise, and how to manage them.
Society’s response, despite promising first steps, is incommensurate with the possibility of rapid, transformative progress that is expected by many experts. AI safety research is lagging. Present governance initiatives lack the mechanisms and institutions to prevent misuse and recklessness, and barely address autonomous systems.

In this short consensus paper, we describe extreme risks from upcoming, advanced AI systems. Drawing on lessons learned from other safety-critical technologies, we then outline a comprehensive plan combining technical research and development (R&D) with proactive, adaptive governance mechanisms for a more commensurate preparation.

Paper : [https://arxiv.org/abs/2310.17688](https://arxiv.org/abs/2310.17688) - Managing extreme AI risks amid rapid progress

# Areas of research

## Oversight and honesty

Capable AI systems can better exploit weaknesses in **technical oversight** and **testing,** for example, by producing false but compelling outputs.

1. A. Pan, K. Bhatia, and J. Steinhardt, “The effects of **[reward misspecification]()**: Mapping and mitigating misaligned models,” International Conference on Learning Representations, 2022.
   Abstract - Reward hacking—where RL agents exploit gaps in misspecified reward functions—has been widely observed, but not yet systematically studied. To understand how reward hacking arises, we construct four RL environments with misspecified rewards. We investigate reward hacking as a function of agent capabilities: model capacity, action space resolution, observation space noise, and training time. More capable agents often exploit reward misspecifications, achieving higher proxyrewardand lowertruerewardthanless capableagents. Moreover, we find instances of phase transitions: capability thresholds at which the agent’s behavior qualitatively shifts, leading to a sharp decrease in the true reward. Such phase transitions pose challenges to monitoring the safety of ML systems. To address this, we propose an anomaly detection task for aberrant policies and offer several baseline detectors.
2. Open problems and fundamental limitations of reinforcement learning from human feedback,” Jul. 2023. arXiv: 2307.1 5217 [cs.AI].
   Abstract - **[Reinforcement learning from human feedback (RLHF)]()** is a technique for training AI systems to align with human goals. RLHF has emerged as the central method used to finetune stateof-the-art large language models (LLMs). Despite this popularity, there has been relatively little public work systematizing its flaws. In this paper, we (1) survey open problems and fundamental limitations of RLHF and related methods; (2) overview techniques to understand, improve, and complement RLHF in practice; and (3) propose auditing and disclosure standards to improve societal oversight of RLHF systems. Our work emphasizes the limitations of RLHF and highlights the importance of a multi-layered approach to the development of safer AI systems.
3. D. Hendrycks, N. Carlini, J. Schulman, and J. Steinhardt, “**[Unsolved problems in ML safety](),**” Sep. 2021. arXiv: 2109 . 13916 [cs.LG].
   robustness, monitoring, aligning, systemic safety
   1. Robustness
   2. Monitoring
   3. Alignment
   4. Systemic safety
4. S. Zhuang and D. Hadfield-Menell, “**[Consequences of misaligned AI]()**,” Adv. Neural Inf. Process. Syst., vol. 33, pp. 15 763–15 773, 2020.
   we should view the design of reward functions as an interactive and dynamic process and identifies a theoretical scenario where some degree of interactivity is desirable.
5. L. Gao, J. Schulman, and J. Hilton, “**[Scaling laws for reward model overoptimization]()**,”in Proceedings of the 40th International Conference on Machine Learning, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., ser. Proceedings of Machine Learning Research, vol. 202, PMLR, 2023, pp. 10 835–10 866.
6. Towards understanding [**sycophancy in language models**](),”Oct. 2023. arXiv: 2310.13548 [cs.CL].
7. D. Amodei, P. Christiano, and A. Ray, [**Learning from human preferences**](https://arxiv.org/abs/1706.03741), https://open ai.com/research/learning-from-human-preferences, Accessed: 2023-9-15,n.d.

## Robustness

AI systems behave unpredictably in new situations. While some aspects of robustness improve with model scale50, other aspects do not or even get worse44,51–53.

1. [**Unsolved problems in ML safety**]()
2. L. L. D. Langosco, J. Koch, L. D. Sharkey, J. Pfau, and D. Krueger, “[**Goal misgeneralization in deep reinforcement learning**](),”Proceedings of Machine Learning Research, vol. 162, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, Eds., pp. 12 004–12 019, 2022.
3. R. Shah, V. Varma, R. Kumar, M. Phuong, V. Krakovna, J. Uesato, and Z. Kenton, “[**Goal misgeneralization: Why correct specifications aren**]()’t enough for correct goals,” Oct. 2022. arXiv: 2210.01790 [cs.LG].
4. T. T. Wang, A. Gleave, T. Tseng, K. Pelrine, N. Belrose, J. Miller, M. D. Dennis, Y. Duan, V. Pogrebniak, S. Levine, and S. Russell, “[**Adversarial policies beat superhuman go AIs**](),”Nov. 2022. arXiv: 2211.00241 [cs.LG].

## Interpretability and transparency

Interpretability and transparency: AI decisionmaking is opaque, with larger, more capable, models being more complex to interpret. So far, we can only test large models via trial and error. We need to learn to understand their inner workings54.

1. T. R¨auker, A. Ho, S. Casper, and D. Hadfield- Menell, “**[Toward transparent AI]():** A survey on interpreting the inner structures of deep neural networks,” in 2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), Feb. 2023, pp. 464–483.


## Evaluation for dangerous capabilities

As AI developers scale their systems, unforeseen capabilities appear spontaneously, without explicit programming59. They are often only discovered after deployment60–62. We need rigorous methods to elicit and assess AI capabilities, and to predict them before training. This includes both generic capabilities
to achieve ambitious goals in the world (e.g., long-term planning and execution), as well as specific dangerous capabilities based on threat models (e.g. social manipulation or hacking). Current evaluations of frontier AI models for dangerous capabilities63— key to various AI policy frameworks—are limited to spot-checks and attempted demonstrations in specific settings36,64,65. These evaluations can sometimes demonstrate dangerous capabilities but cannot reliably rule them out: AI systems that lacked certain capabilities in the tests may well demonstrate them in slightly different settings or with post-training enhancements. Decisions that depend on AI systems not crossing any red lines thus need large safety margins. Improved evaluation tools decrease the chance of missing dangerous capabilities, allowing for smaller margins.

1. J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, E. H. Chi, T. Hashimoto, O. Vinyals, P. Liang, J. Dean, and W. Fedus, “[**Emergent abilities of large language models**](),”Transactions on Machine Learning Research, Jun. 2022.
2. J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, and D. Zhou, [**Chain-of-thought prompting**]() elicits reasoning in large language models,” Adv. Neural Inf. Process. Syst., vol. 35, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., pp. 24 824–24 837, Jan. 2022.
3. P. Zhou, J. Pujara, X. Ren, X. Chen, H.-T. Cheng, Q. V. Le, E. H. Chi, D. Zhou, S. Mishra, and H. S. Zheng, “[**Self-Discover: Large language models self-compose reasoning structures**](),” Feb. 2024. arXiv: 2402 .03620 [cs.AI].
4. T. Davidson, J.-S. Denain, P. Villalobos, and G. Bas, “**[AI capabilities can be significantly improved without expensive retraining](),**”Dec. 2023. arXiv: 2312 . 07413
   [cs.AI].
5. T. Shevlane, S. Farquhar, B. Garfinkel, M. Phuong, J. Whittlestone, J. Leung, D. Kokotajlo, N. Marchal, M. Anderljung, N. Kolt, L. Ho, D. Siddarth, S. Avin, W. Hawkins, B. Kim, I. Gabriel, V. Bolina, J. Clark, Y. Bengio, P. Christiano, and A. Dafoe, “[**Model evaluation for extreme risks**](),” May 2023. arXiv: 2305.15324 [cs.AI].
6. C. A. Mouton, C. Lucas, and E. Guest, The operational risks of AI in large-scale biological attacks: Results of a red-team study, Santa Monica, CA, 2024.
7. J. Scheurer, M. Balesni, and M. Hobbhahn,“Technical report: Large language models can [**strategically deceive their users when put under pressure**](),” Nov. 2023. arXiv: 231 1.07590 [cs.CL].
8. M. Kinniment, L. J. K. Sato, H. Du, B. Goodrich, M. Hasin, L. Chan, L. H. Miles, T. R. Lin, H. Wijk, J. Burget, A. Ho, E. Barnes, and P. Christiano, “[**Evaluating language-model agents on realistic autonomous tasks**](),”
9. [**Goodhard Law of reinforcement learning**](https://arxiv.org/abs/2310.09144) - 

## Evaluating AI alignment

If AI progress continues, AI systems will eventually possess highly dangerous capabilities. Before training and deploying such systems, we need methods to assess their propensity to use these capabilities. Purely behavioral evaluations may fail for advanced AI systems: like humans, they might behave differently under evaluation, faking alignment56–58.

1. R. Ngo, L. Chan, and S. Mindermann, “[**The alignment problem from a deep learning perspective**](),” International Conference on Learning Representations 2024, Jan. 2024
2. Sleeper agents: [**Training deceptive LLMs** ]()that persist through safety training, Jan. 2024. arXiv: 2401.05566 [cs.CR].
3. M. K. Cohen, N. Kolt, Y. Bengio, G. K. Hadfield, and S. Russell, “[**Regulating advanced artificial agents**,]()” Science, vol. 384, no. 6691, pp. 36–38, Apr. 2024.


## Interpretability and transparency

AI decisionmaking is opaque, with larger, more capable, models being more complex to interpret. So far, we can only test large models via trial and error. We need to learn to understand their inner workings54.

1. T. R¨auker, A. Ho, S. Casper, and D. Hadfield-Menell, “[**Toward transparent AI: A survey on interpreting**]() the inner structures of deep neural networks,” in 2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), Feb. 2023, pp. 464–483.

## Resilience

Inevitably, some will misuse or act recklessly with AI. We need tools to detect and defend against AI-enabled threats such as large-scale influence operations, biological risks, and cyberattacks. However, as AI systems become more capable, they will eventually be able to circumvent human-made defenses. To enable more powerful AI-based defenses, we first need to learn how to make AI systems safe and aligned.

1. D. Hendrycks, N. Carlini, J. Schulman, and J. Steinhardt, “**[Unsolved problems in ML safety](),**” Sep. 2021. arXiv: 2109 . 13916 [cs.LG].
3. A. Ho and J. Taylor, [**Using advance market commitments**]() for public purpose technology development, Policy Brief, Jun. 2021.
