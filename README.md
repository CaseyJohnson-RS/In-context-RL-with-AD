# In-Context Reinforcement Learning with Algorithm Distillation

This repository is an environment for running experiments with algorithm distillation. Link to work in this area: [arxiv](https://arxiv.org/abs/2210.14215).

---

## How to run the project?

| Tool | Version |
| ------------- | ----------------------------------- |
| Python | Python 3.10 |
| CUDA Toolkit | 12.8 |
| nvcc Compiler | 12.8.61 |
| CUDA Build | cuda_12.8.r12.8/compiler.35404655_0 |

Installing the CUDA toolchain is optional, and I'll leave that for you to explore.

###### 1. Create a virtual environment

If you only have one Python interpreter installed, you can create a virtual environment with the command:
```bash
python -m venv <your_env_name>
```

If you have multiple Python versions, you can select the one you need with the command:
```bash
py -3.10 -m venv <your_env_name>
```

*Make sure you create the environment in the correct directory.*

###### 2. Activate the environment

Windows
```cmd
<your_env_name>\Scripts\activate.bat
```

Linux
```bash
source <your_env_name>/bin/activate
```

###### 3. Install libraries

I'd like to point out that the virtual environment in the total size is about 6 GB. This isn't a warning, just a fact. Be prepared.

```bash
pip install -r requirements.txt
```

###### 4. Starting the MLFlow server

Create a directory in advance to store the MLFlow server data.

```bash
<mlflow_dir_path>
├── data_local/
└── artifacts/
```

Next, run the command.

```bash
mlflow server \
--backend-store-uri "file:///<mlflow_dir_abspath>/data_local" \
--default-artifact-root "file:///<mlflow_dir_abspath>/artifacts" \
--host localhost \
--port 5000
```

Now we have a local server running on port 5000. You can check [http://localhost:5000](http://localhost:5000).

###### 5. Run the experiment

Go to the `scripts` folder. It contains directories with the names of the environments on which the experiments were run. Each directory contains scripts for training and evaluating models (to change hyperparameters, you need to change the values ​​inside the script).

```bash
python scripts/<experiment_name>/<experiment_script>.py
```