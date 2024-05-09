# CPSC477 Final Project

This repository contains the code for the final project of CPSC477: Natural Language Processing.

The project tests two LLM defenses: SmoothLLM and Erase-and-Check, in order to determine their effectiveness in defending against adversarial attacks. Our project utilised a single L4 GPU instance on Google Cloud Platform to run the attacks.

The project also tests the effectiveness of the attacks themselves.
## Structure
The project is structured as follows:

- `certifiedllmsafety/` is based on the repository from Kumar et al. (2024) which contains the code for the SmoothLLM and Erase-and-Check defenses.
- `JailbreakingLLMs/` is based on the repository from Chao et al. (2024) which contains the code for PAIR attacks.
- `llm-attacks/` is based on the repository from Zou et al. (2023) which contains the code for the GCG attacks.
- `results/` is the output folder for the experiment results.
- `smoothllm/` is based on the repository from Robey et al. (2023) which contains the code for the SmoothLLM defense.

## Set up and running

To run the project, the llama model weights must be stored at `DIR/llama-2/llama/llama-2-7b-chat-hf`. These weights can be obtained here: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

Be sure to install the required packages by running `pip install -r requirements.txt`

To run the project and various experiments as described in our final report, run the following commands:

```
python main.py --trial=0 --attack=GCG --defense=NONE
python main.py --trial=1 --attack=PAIR --defense=NONE

python main.py --trial=2 --attack=SAFE --defense=NONE
python main.py --trial=3 --attack=SAFE --defense=NONE

python main.py --trial=4 --attack=SAFE --defense=EC --ec_type=smoothing --max_erase=20
python main.py --trial=5 --attack=SAFE --smoothllm_num_copies=2 --smoothllm_pert_pct=5 --smoothllm_pert_type=RandomSwapPerturbation --defense=SmoothLLM

python main.py --trial=6 --attack=GCG --defense=EC --ec_type=smoothing --max_erase=20
python main.py --trial=7 --attack=GCG --smoothllm_num_copies=2 --smoothllm_pert_pct=5 --smoothllm_pert_type=RandomSwapPerturbation --defense=SmoothLLM

python main.py --trial=8 --attack=PAIR --defense=EC --ec_type=smoothing --max_erase=20
python main.py --trial=9 --attack=PAIR --smoothllm_num_copies=2 --smoothllm_pert_pct=5 --smoothllm_pert_type=RandomSwapPerturbation --defense=SmoothLLM

python main.py --trial=10 --attack=SAFE --defense=EC --ec_type=smoothing --max_erase=4
```

Most possible permutations of experiments are in the commands.txt file.

## Our Work

Some noticeable changes to the original code from the original authors include:

- creating wrappers for the SmoothLLM and Erase-and-Check defenses to allow for easier integration with the attack code, see: `smoothLLM/lib/defenses.py`
- writing a main.py script for running the experiments, which involved integrating the different wrapped LLM instances with the attack code, and also writing an automated attack script for the PAIR attack see: `main.py`
- modifying the attack code to allow for the PAIR attack to be run using the same model for both the attack and target model
- demonstrating the plug-and-play capability with gpt-3.5-turbo in the `gpt_experiments.ipynb` notebook

## Credit

The project was completed by:

- Bryan Wee
- Feranmi Oluwadairo
- Liam Varela
