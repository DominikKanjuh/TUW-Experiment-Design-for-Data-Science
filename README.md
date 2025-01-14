# TUW-Experiment-Design-for-Data-Science

Reproduction of experiments from Zero-Shot Recommendation as Language Modeling by Damien Sileo, Wout Vossen and Robbe Raymaekers, exploring the use of pretrained language models for recommendation without structured training data. Part of TU Wien's Experiment Design for Data Science course - 188.992 (WS 2024/25)

## How to run the code

Firstly go on the [Weights & Biases dashboard](https://wandb.ai/site/) and create a new project. Copy the wandb API key and paste it .env file which you can create by running:

```bash
cp .env.example .env
```

After that install Docker on your machine and run the following command to install the dependencies:

```bash
docker build -t gpt-rec .
```

Run the container with the following command:

```bash
docker run --env-file .env gpt-rec
```
