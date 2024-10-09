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

Paper : [https://arxiv.org/abs/2310.17688](https://arxiv.org/abs/2310.17688)

# Areas of research

## Oversight and honesty

Capable AI systems can better exploit weaknesses in **technical oversight** and **testing,** for example, by producing false but compelling outputs. 

1. A. Pan, K. Bhatia, and J. Steinhardt, “The effects of reward misspecification: Mapping and mitigating misaligned models,” International Conference on Learning Representations, 2022.
   Abstract - Reward hacking—where RL agents exploit gaps in misspecified reward functions—has been widely observed, but not yet systematically studied. To understand how reward hacking arises, we construct four RL environments with misspecified rewards. We investigate reward hacking as a function of agent capabilities: model capacity, action space resolution, observation space noise, and training time. More capable agents often exploit reward misspecifications, achieving higher proxyrewardand lowertruerewardthanless capableagents. Moreover, we find instances of phase transitions: capability thresholds at which the agent’s behavior qualitatively shifts, leading to a sharp decrease in the true reward. Such phase transitions pose challenges to monitoring the safety of ML systems. To address this, we propose an anomaly detection task for aberrant policies and offer several baseline detectors.
2. Open problems and fundamental limitations of reinforcement learning from human feedback,” Jul. 2023. arXiv: 2307.1 5217 [cs.AI].
   Abstract - Reinforcement learning from human feedback (RLHF) is a technique for training AI systems to align with human goals. RLHF has emerged as the central method used to finetune stateof-the-art large language models (LLMs). Despite this popularity, there has been relatively little public work systematizing its flaws. In this paper, we (1) survey open problems and fundamental limitations of RLHF and related methods; (2) overview techniques to understand, improve, and complement RLHF in practice; and (3) propose auditing and disclosure standards to improve societal oversight of RLHF systems. Our work emphasizes the limitations of RLHF and highlights the importance of a multi-layered approach to the development of safer AI systems.
3. D. Hendrycks, N. Carlini, J. Schulman, and J. Steinhardt, “Unsolved problems in ML safety,” Sep. 2021. arXiv: 2109 . 13916 [cs.LG].
   robustness, monitoring, aligning, systemic safety
4. S. Zhuang and D. Hadfield-Menell, “Consequences of misaligned AI,” Adv. Neural Inf. Process. Syst., vol. 33, pp. 15 763–15 773, 2020.
   we should view the design of reward functions as an interactive and dynamic process and identifies a theoretical scenario where some degree of interactivity is desirable.
5. L. Gao, J. Schulman, and J. Hilton, “Scaling laws for reward model overoptimization,”in Proceedings of the 40th International Conference on Machine Learning, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., ser. Proceedings of Machine Learning Research, vol. 202, PMLR, 2023, pp. 10 835–10 866.
6. Towards understanding sycophancy in language models,”Oct. 2023. arXiv: 2310.13548 [cs.CL].
7. D. Amodei, P. Christiano, and A. Ray, Learning from human preferences, https://open ai.com/research/learning-from-human-preferences, Accessed: 2023-9-15,n.d.

## Robustness 

AI systems behave unpredictably in new situations. While some aspects of robustness improve with model scale50, other aspects do not or even get worse44,51–53.

## Interpretability and transparency

Interpretability and transparency: AI decisionmaking is opaque, with larger, more capable, models being more complex to interpret. So far, we can only test large models via trial and error. We need to learn to understand their inner workings54.

## Evaluation for dangerous capabilities

As AI developers scale their systems, unforeseen capabilities appear spontaneously, without explicit programming59. They are often only discovered after deployment60–62. We need rigorous methods to elicit and assess AI capabilities, and to predict them before training. This includes both generic capabilities
to achieve ambitious goals in the world (e.g., long-term planning and execution), as well as specific dangerous capabilities based on threat models (e.g. social manipulation or hacking). Current evaluations of frontier AI models for dangerous capabilities63— key to various AI policy frameworks—are limited to spot-checks and attempted demonstrations in specific settings36,64,65. These evaluations can sometimes demonstrate dangerous capabilities but cannot reliably rule them out: AI systems that lacked certain capabilities in the tests may well demonstrate them in slightly different settings or with post-training enhancements. Decisions that depend on AI systems not crossing any red lines thus need large safety margins. Improved evaluation tools decrease the chance of missing dangerous capabilities, allowing for smaller margins.

## Evaluating AI alignment

If AI progress continues, AI systems will eventually possess highly dangerous capabilities. Before training and deploying such systems, we need methods to assess their propensity to use these capabilities. Purely behavioral evaluations may fail for advanced AI systems: like humans, they might behave differently under evaluation, faking alignment56–58.

## Interpretability and transparency

AI decisionmaking is opaque, with larger, more capable, models being more complex to interpret. So far, we can only test large models via trial and error. We need to learn to understand their inner workings54.

## Resilience 

Inevitably, some will misuse or act recklessly with AI. We need tools to detect and defend against AI-enabled threats such as large-scale influence operations, biological risks, and cyberattacks. However, as AI systems become more capable, they will eventually be able to circumvent human-made defenses. To enable more powerful AI-based defenses, we first need to learn how to make AI systems safe and aligned. 

1. Unsolved problems in ML safety,” Sep. 2021. arXiv: 2109 . 13916 [cs.LG].
