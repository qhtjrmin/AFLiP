from __future__ import annotations

import argparse
import copy
import os
import random
import time
from typing import Iterable

import numpy as np
import torch
from ogb.linkproppred import Evaluator

from prepare_dataset import loaddataset
import flip_lp as aflip
from utils.early_stopping import EarlyStopping


LARGE_DATASETS = {"citation2", "twitter", "friendster"}
CITATION_DATASETS = {"Cora", "Citeseer", "Pubmed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Implementation of AFLiP"
    )

    # Dataset / runtime
    parser.add_argument("--dataset", type=str, default="collab")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--log_steps", type=int, default=1)

    # Model
    parser.add_argument("--model", type=str, default="SAGE")
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--decoder", type=str, default="mlp")
    parser.add_argument("--mask_input", action="store_true")

    # Optimization
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # Decoder options used by NCN decoder
    parser.add_argument("--cndeg", type=int, default=-1)
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--beta", type=float, default=1.0)

    # Experiment modes
    parser.add_argument(
        "--train_mode",
        type=str,
        default=None,
        help="Single training mode, e.g., full, sampling, static, aflip",
    )
    parser.add_argument(
        "--train_modes",
        type=str,
        default=None,
        help="Comma-separated training modes, e.g., full,static,aflip",
    )
    parser.add_argument("--fanout", type=str, default="20,15,10")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def get_train_modes(args: argparse.Namespace) -> list[str]:
    if args.train_modes:
        return [mode.strip() for mode in args.train_modes.split(",") if mode.strip()]
    if args.train_mode:
        return [args.train_mode]
    return ["full", "sampling", "static", "adaptive"]


def get_metric_name(dataset: str) -> str:
    if dataset == "collab":
        return "Hits@50"
    if dataset == "citation2":
        return "MRR"
    return "Hits@100"


def get_evaluator(dataset: str) -> Evaluator:
    if dataset in CITATION_DATASETS:
        return Evaluator(name="ogbl-ppa")
    if dataset in {"twitter", "friendster"}:
        return Evaluator(name="ogbl-citation2")
    return Evaluator(name=f"ogbl-{dataset}")


def move_data_to_device(data, dataset: str, device: torch.device):
    if dataset not in LARGE_DATASETS:
        return data.to(device)

    data.x = data.x.to(device)
    if hasattr(data, "ptr"):
        data.ptr = data.ptr.to(device)
    if hasattr(data, "idx"):
        data.idx = data.idx.to(device)
    return data


def build_models(args: argparse.Namespace, data, device: torch.device):
    """Build encoder and link predictor."""
    if args.train_mode == "sampling":
        encoder = aflip.get_encoder_sampling(args, data.num_features, data=data).to(device)
    else:
        encoder = aflip.get_encoder(args, data.num_features, data=data).to(device)

    predictor_layers = args.num_layers if args.dataset in CITATION_DATASETS else 2
    predictor = aflip.build_decoder(args, predictor_layers, data=data).to(device)
    return encoder, predictor


def build_history(args: argparse.Namespace, data, train_mode: str, device: torch.device):
    """Build history cache used by static/aflip modes."""
    if train_mode not in {"static", "aflip"} and "aflip" not in train_mode:
        return None

    input_dim = args.hidden_channels if getattr(data, "max_x", -1) > -1 else data.input_dim
    return aflip.MultiLayerFullCache(
        num_nodes=data.num_nodes,
        input_dim=input_dim,
        hidden_dim=args.hidden_channels,
        num_layers=args.num_layers,
        device=device,
        model=args.model,
    )


def build_adaptive_state(args: argparse.Namespace, data, train_mode: str, device: torch.device):
    """State used by aflip modes."""
    if "aflip" not in train_mode:
        return None

    return aflip.AdaptiveState(
        num_nodes=data.num_nodes,
        num_layers=args.num_layers,
        use_grad_staleness=True,
        device=device,
        topk_ratio=0.1,
        momentum=0.8,
    )


def evaluate(
    encoder,
    predictor,
    data,
    split_edge,
    evaluator,
    args: argparse.Namespace,
    train_mode: str,
):
    test_fn = aflip.test_citation_full if args.dataset == "citation2" else aflip.test_collab_full
    return test_fn(
        encoder,
        predictor,
        data,
        split_edge,
        evaluator,
        args.batch_size,
        train_mode,
    )


def train_one_mode(
    args: argparse.Namespace,
    data,
    split_edge,
    evaluator,
    train_mode: str,
    init_encoder_state: dict,
    init_predictor_state: dict,
    device: torch.device,
) -> tuple[float, float, int]:
    """Train one mode and return best validation score, test score, and epoch."""
    args.train_mode = train_mode

    encoder, predictor = build_models(args, data, device)
    encoder.load_state_dict(copy.deepcopy(init_encoder_state))
    predictor.load_state_dict(copy.deepcopy(init_predictor_state))

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metric_name = get_metric_name(args.dataset)
    early_stopper = EarlyStopping(patience=10, min_delta=0.0, mode="max")

    history = build_history(args, data, train_mode, device)
    adaptive_state = build_adaptive_state(args, data, train_mode, device)

    best_valid, best_test, best_epoch = float("-inf"), float("-inf"), 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        loss = aflip.run_train_epoch(
            args=args,
            encoder=encoder,
            predictor=predictor,
            data=data,
            split_edge=split_edge,
            optimizer=optimizer,
            train_mode=train_mode,
            history=history,
            state=adaptive_state,
            analyzer=None,
        )

        print(
            f"[{train_mode}] epoch={epoch:03d} "
            f"loss={loss:.4f} train_time={time.time() - start:.2f}s"
        )

        if epoch % args.log_steps != 0:
            continue

        raw_results = evaluate(
            encoder, predictor, data, split_edge, evaluator, args, train_mode
        )
        results = raw_results if isinstance(raw_results, dict) else {metric_name: raw_results}
        train_metric, valid_metric, test_metric = results[metric_name]

        print(
            f"[{train_mode}] epoch={epoch:03d} "
            f"train={train_metric:.4f} valid={valid_metric:.4f} test={test_metric:.4f}"
        )

        if valid_metric > best_valid:
            best_valid = valid_metric
            best_test = test_metric
            best_epoch = epoch

        if early_stopper.step(valid_metric, epoch, encoder, predictor):
            print(f"[{train_mode}] early stopping at epoch {epoch}")
            break

    return best_valid, best_test, best_epoch


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )

    train_modes = get_train_modes(args)
    evaluator = get_evaluator(args.dataset)

    data, split_edge = loaddataset(args.dataset, use_valedges_as_input=False)
    data = move_data_to_device(data, args.dataset, device)

    print(f"Dataset: {args.dataset}")
    print(f"Train modes: {train_modes}")
    print(f"Device: {device}")

    all_results = {}

    for run in range(args.num_runs):
        run_seed = args.seed + run
        set_seed(run_seed)

        print(f"\n========== Run {run + 1}/{args.num_runs}, seed={run_seed} ==========")

        # Build one initial model per run
        args.train_mode = train_modes[0]
        encoder0, predictor0 = build_models(args, data, device)
        init_encoder_state = copy.deepcopy(encoder0.state_dict())
        init_predictor_state = copy.deepcopy(predictor0.state_dict())

        for train_mode in train_modes:
            valid, test, epoch = train_one_mode(
                args=args,
                data=data,
                split_edge=split_edge,
                evaluator=evaluator,
                train_mode=train_mode,
                init_encoder_state=init_encoder_state,
                init_predictor_state=init_predictor_state,
                device=device,
            )

            all_results.setdefault(train_mode, []).append((valid, test, epoch))

    print("\n========== Summary ==========")
    for train_mode, values in all_results.items():
        values = np.asarray(values, dtype=float)
        valid_mean, valid_std = values[:, 0].mean(), values[:, 0].std()
        test_mean, test_std = values[:, 1].mean(), values[:, 1].std()
        epoch_mean = values[:, 2].mean()

        print(
            f"{train_mode}: "
            f"valid={valid_mean:.4f}±{valid_std:.4f}, "
            f"test={test_mean:.4f}±{test_std:.4f}, "
            f"avg_epoch={epoch_mean:.1f}"
        )


if __name__ == "__main__":
    main()
