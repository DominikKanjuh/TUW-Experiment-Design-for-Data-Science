# https://huggingface.co/datasets/sileod/movie_recommendation

from xpflow import Xp
import bigbench.models.huggingface_models as huggingface_models
from tensorflow.keras import mixed_precision
import pandas as pd
import numpy as np
import random
import torch
from tqdm.auto import tqdm
import sklearn
from easydict import EasyDict as edict
from collections import defaultdict
from itertools import chain
import wget
import zipfile
import os
import wandb
from appdirs import user_data_dir
import pathlib
from dotenv import load_dotenv


def precision_recall(y_true, y_pred, k):
    nz = pd.DataFrame(y_true.nonzero()).T
    nz.columns = ["user", "item"]
    nz = np.array(list(nz.groupby("user")["item"].agg(list)))

    precision, recall = [], []
    for true, pred in zip(nz, (-y_pred).argsort(axis=1)[:, :k]):
        u_recall = np.mean([x in pred for x in true])
        u_precision = np.mean([x in true for x in pred])
        precision += [u_precision]
        recall += [u_recall]
    return {f"precision_{k}": np.mean(precision), f"recall_{k}": np.mean(recall)}


def make_metrics(y_true, y_pred):
    metrics = defaultdict(list)
    for k in [1, 2, 3, 4, 5]:
        for i in range(len(y_true)):
            yt, yp = y_true[[i], :], y_pred[[i], :]
            metrics[f"ndcg_{k}"] += [
                sklearn.metrics.ndcg_score(y_true=yt, y_score=yp, k=k)
            ]
            metrics[f"precision_{k}"] += [precision_recall(yt, yp, k)[f"precision_{k}"]]
            metrics[f"recall_{k}"] += [precision_recall(yt, yp, k)[f"recall_{k}"]]

    for m in list(metrics.keys()):
        metrics[f"{m}_std"] = np.std(metrics[m])
        metrics[m] = np.mean(metrics[m])

    return dict(metrics)


def make_pop(y_pred, y_pops):
    pop_1 = []
    for pops, i in zip(y_pops, y_pred.argmax(axis=1)):
        pop_1 += [pops[i]]
    return {"pop_1": np.mean(pop_1), "pop_1_std": np.std(pop_1)}


# Build data

root = pathlib.Path(user_data_dir("gpt-rec"))
root.mkdir(exist_ok=True)
os.chdir(root)
url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

if not os.path.exists("ml-1m"):
    filename = wget.download(url)
    zipfile.ZipFile(filename).extractall()
os.chdir("ml-1m")


def process_movielens_name(s):
    s = s[:-7]
    s = s.split(" (")[0]
    for pattern in [", The", ", A"]:
        if s.endswith(pattern):
            s = pattern.split(", ")[1] + " " + s.replace(pattern, "")
    return s


items = pd.read_csv(
    "movies.dat",
    sep="::",
    names=["movieId", "title", "genres"],
    engine="python",
    encoding="latin-1",
)
items["name"] = items.title.map(process_movielens_name)
item_id_to_name = items.set_index("movieId")["name"].to_dict()

prompts = (
    "[M]",  # 0
    "Movies like [M]",  # 1
    "Movies similar to [M]",  # 2
    "Movies like: [M]",  # 3
    "Movies similar to: [M]",  # 4
    "If you liked [M] you will also like",
)  # 5


def make_prompt(l, xp):
    movies = xp.sep.join(random.sample([item_id_to_name[i] for i in l], xp.nb_pos))
    prompt = prompts[xp.prompt_id].replace("[M]", movies)
    return prompt + xp.end_sep


def make_data(xp):
    df = pd.read_csv(
        "ratings.dat",
        sep="::",
        names=["userId", "movieId", "rating", "ts"],
        engine="python",
    )
    df = df[~df.rating.between(2.4, 4.1)]
    # R = df.pivot('movieId','userId','rating')
    R = df.pivot_table(index="movieId", columns="userId", values="rating")

    pos_neg = (
        df.groupby("userId")["movieId"]
        .agg(list)
        .reset_index()
        .sample(frac=1.0, random_state=xp.users_seed)
    )
    pos_neg["pos"] = pos_neg.apply(
        lambda x: [i for i in x.movieId if R[x.userId][i] > xp.like_threshold], axis=1
    )
    pos_neg["neg"] = pos_neg.apply(
        lambda x: [i for i in x.movieId if R[x.userId][i] < xp.dislike_threshold],
        axis=1,
    )
    pos_neg = pos_neg.set_index("userId")
    pos_neg = pos_neg[pos_neg.pos.map(len).ge(xp.min_pos_ratings)]
    pos_neg = pos_neg[pos_neg.neg.map(len).ge(xp.min_neg_ratings)]

    pos_neg["support"] = pos_neg.pos.map(lambda x: random.sample(x, xp.nb_pos + 1))
    pos_neg["targets"] = pos_neg.support.map(lambda x: [x[-1]]) + pos_neg.neg.map(
        lambda x: random.sample(x, xp.nb_neg)
    )
    pos_neg["support"] = pos_neg["support"].map(lambda x: x[:-1])
    pos_neg["choices"] = pos_neg.targets.map(
        lambda l: tuple([item_id_to_name[i] for i in l])
    )
    pos_neg["prompt"] = pos_neg.support.map(lambda l: make_prompt(l, xp))

    pop = (R.sum(axis=1) / (R.T.sum().mean())).to_dict()
    pos_neg["pop"] = pos_neg.targets.map(lambda l: [pop[x] for x in l])

    return pos_neg


from transformers import BertForPreTraining, AutoTokenizer


class BERT:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        self.model = BertForPreTraining.from_pretrained(args.model_type)
        self._model = edict(_model_name=args.model_type)

    def cond_log_prob(self, inputs, targets):
        inputs, targets = [inputs] * len(targets), list(targets)
        scores = self.model(
            **self.tokenizer(inputs, targets, return_tensors="pt", padding=True)
        ).seq_relationship_logits
        return list(scores[:, 0].cpu().detach().numpy())


# Experiment

if torch.cuda.is_available():
    mixed_precision.set_global_policy("mixed_float16")

if torch.cuda.is_available():
    mixed_precision.set_global_policy("mixed_float16")


class base(Xp):
    like_threshold = 4
    dislike_threshold = 2.5
    min_pos_ratings = 21
    min_neg_ratings = 5
    nb_pos = 5
    nb_neg = 4
    prompts = str(prompts)
    prompt_id = [0, 3]
    data_path = os.getcwd()
    model_type = "gpt2"
    nb_test_users = 50
    users_seed = 0
    sep = ","
    end_sep = ","
    offset = [0, 50, 100, 150]


class model_size(base):
    model_type = ["gpt2", "gpt2-medium", "gpt2-large"][::-1]


class nb_pos(base):
    nb_pos = [1, 2, 3, 5, 7, 10, 15, 20]


class prompts_types(base):
    prompt_id = [0, 1, 2, 3, 4, 5]
    # sep=[', ','\n']


class penha(base):
    prompt_id = 5
    end_sep = " "
    model_type = ["bert-base-uncased", "bert-large-uncased"]


# Load environment variables from .env file and login to wandb
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))


for xp in tqdm(
    list(
        chain(
            *[
                x()
                for x in [
                    nb_pos,
                ]
            ]
        )
    )
):
    xp_hash = f"{hash(xp)}.txt"
    #  if xp_hash in {x.name for x in dbx.files_list_folder('/colab/log').entries}:
    #    continue

    #  run = wandb.init(project='gpt-rec', entity='',reinit=True, config=xp);
    with wandb.init(
        project="gpt-rec",
        entity="",
        reinit=True,
        config=xp,
    ) as run:
        pos_neg = make_data(xp)
        if "bert" in xp.model_type:
            model = BERT(xp)
        else:
            model = huggingface_models.BIGBenchHFModel(xp.model_type)
        l = []
        users = list(range(xp.offset, xp.offset + xp.nb_test_users))
        for i in tqdm(users):
            scores = model.cond_log_prob(
                inputs=list(pos_neg.prompt)[i], targets=list(pos_neg.choices)[i]
            )
            l += [scores]
        y_pred = np.array(l)
        y_true = y_pred * 0
        y_true[:, 0] = 1
        xp.result = make_metrics(y_true, y_pred)
        wandb.log(xp.result)
        wandb.log(make_pop(y_pred, pos_neg.iloc[users]["pop"]))
        run.finish()
