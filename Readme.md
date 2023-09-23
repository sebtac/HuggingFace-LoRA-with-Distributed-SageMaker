# MAIN OBJECTIVE: 
Implement and test efficiency of LoRA and QLoRA for fine-tuning of LLM models in DDP and MDP distributed modes in SageMaker
           
# SECONDARY OBJECTIVES: 
- Test relative performance and applicability of p3 and g5 EC2 instances to LLM fine-tuning
- Test the impact of LEARNING RATE on training performance in distributed modes
- Test the ability to increase the Batch Size with LoRA and QLoRA in distributed modes


# MOTIVATION:
- The success of LLM models is rooted in HUGE sizes of the models trained on HUGE number of examples on MANY GPUs over LONG period of times resulting in HUGE TRAINIG COST. Thus, training LLMs from-scratch is beyond reach to many individual businesses due to technical and cost restrictions.
- Recently, research community came up with (x)LoRA approaches to train LLM with reduced hardware requirements. Today, a 65B parameter model can be trained on a single GPU with those techniques.
- While we welcome such development, we see a huge potential in combining (x)LoRA training with Data Parallelism for even faster training. In such scenarios:
  - LoRA is the enabler which makes feasible an industrial application of LLMs (reasonable hardware requirements and budget sizes)
  - Data Parallelism provides training speedups that make fine-tuning of LLMs feasible for wide use by industry (reasonable time scales)
- We also explore feasibility of combining (x)LoRA approaches with Model Parallelism to enable efficient training of bigger than 65B models - which we expect to be of huge interest to the industry in the coming years.


# KEY FINDINGS:
- Both LoRA and QLoRA were successfully implemented in DDP mode in SageMaker with expected benefits in:
    - Resource Requirements
    - Training Speed
- LoRA was successfully implemented in MDP mode in SageMaker with expected benefits
- QLoRA was not successfully implemented in MDP mode as training freezes due to unknown reason (no error message is generated and deeper troubleshooting is needed)
- In Single GPU Mode, P3 Instances offer better performance (for smaller models) despite lower GPU memory per instance (P3:16GB vs. G5:24GB)
- G5 instances are not supported in Distributed SageMaker modes due to SageMaker Environment limitations.
- Running LLM Fine-Tuning in distirbuted modes requires boosting Learning Rates more then the ratio of parallelism!
- LoRA and QLoRA allow to increase the Batch Size and require fine-tuning of Learning Rate to achieve comparable training performance to that achived with Single GPU batch size.

- Detailed findings in the file: 


# ANALYSIS:
