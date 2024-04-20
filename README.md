# Large Language Models Powered Context-aware Motion Prediction

[Xiaoji Zheng](https://seu-zxj.github.io), [Lixiu Wu](https://github.com/wuli-maker), [Zhijie Yan](https://github.com/BJHYZJ), [Yuanrong Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang,+Y), [Hao Zhao](https://sites.google.com/view/fromandto), [Chen Zhong](https://arxiv.org/search/cs?searchtype=author&query=Zhong,+C), [Bokui Chen](https://www.sigs.tsinghua.edu.cn/cbk/main.htm), [Jiangtao Gong](https://air.tsinghua.edu.cn/info/1046/1635.htm)

https://github.com/SEU-zxj/LLM-Augmented-MTR/assets/73421144/f0cbbe14-1a9c-496b-87af-89ea78fff087

<video src="https://github.com/SEU-zxj/LLM-Augmented-MTR/assets/73421144/962d7dee-9350-4b0f-abe3-9ac9feab697a"></video>

# Introduction

Motion prediction is among the most fundamental tasks in autonomous driving. Traditional methods of motion forecasting primarily encode vector information of maps and historical trajectory data of traffic participants, lacking a comprehensive understanding of overall traffic semantics, which in turn affects the performance of prediction tasks. In this paper, we utilized Large Language Models (LLMs) to enhance the global traffic context understanding for motion prediction tasks. We first conducted systematic prompt engineering, visualizing complex traffic environments and historical trajectory information of traffic participants into image prompts---Transportation Context Map (TC-Map), accompanied by corresponding text prompts. Through this approach, we obtained rich traffic context information from the LLM. By integrating this information into the motion prediction model, we demonstrate that such context can enhance the accuracy of motion predictions. Furthermore, considering the cost associated with LLMs, we propose a cost-effective deployment strategy: enhancing the accuracy of motion prediction tasks at scale with 0.7% LLM-augmented datasets. Our research offers valuable insights into enhancing the understanding of traffic scenes of LLMs and the motion prediction performance of autonomous driving.

<img src="./fig/main_figure.png" alt="Our idea" style="zoom:30%;" />

# News



# Prompt Engineering

we conducted a comprehensive prompt design and experiment and summarised six prompt suggestions for future researchers who want to utilize LLM's ability to understand BEV-liked complex transportation maps.

![prompt demonstration](./fig/prompt_pages.png)

![prompt design suggestions](./fig/promptDesign.png)

# Get Start

## LLM-Augmented Propmt



## LLM-Augmented MTR



# Acknowledgement

This repository is based on the code from [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset), [MTR](https://github.com/sshaoshuai/MTR)

# Citation

if you find our work are useful, we are happy to hear you cite our work!

```latex
@misc{llm_augmented_mtr,
      title={Large Language Models Powered Context-aware Motion Prediction}, 
      author={Xiaoji Zheng and Lixiu Wu and Zhijie Yan and Yuanrong Tang and Hao Zhao and Chen Zhong and Bokui Chen and Jiangtao Gong},
      year={2024},
      eprint={2403.11057},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
