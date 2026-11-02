import os

print(os.getcwd())
# ----------------------------------------------------------
# .env рядом со скриптом (как в остальных скриптах репо):
# SBEROSC_TOKEN, PATH_OMNI_MAIN, PATH_ROMASHKA, POLARS_MAX_THREADS, RAYON_NUM_THREADS
# ----------------------------------------------------------
with open(".env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key] = value


def _require_env(var_name: str, hint: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(f"Переменная окружения {var_name} не задана ({hint}). Задайте её: export {var_name}=..., либо в .env")
    return value


POLARS_MAX_THREADS = _require_env("POLARS_MAX_THREADS", "")
RAYON_NUM_THREADS = _require_env("RAYON_NUM_THREADS", "")
PATH_OMNI_MAIN = _require_env("PATH_OMNI_MAIN", "/home/datalab/nfs/Omni/omnichannel_main")
PATH_ROMASHKA = _require_env("PATH_ROMASHKA", "/home/datalab/nfs/Omni/Romashka/camomile_models")
SBEROSC_TOKEN = _require_env("SBEROSC_TOKEN", "Токен доступа к SberOSC для доступа к PyPI")

os.system(f"pip install polars==1.29.0 --index-url https://token:{SBEROSC_TOKEN}@sberosc.ca.sbrf.ru/repo/pypi/simple --trusted-host sberosc.ca.sbrf.ru")
os.system(f"pip install torchmetrics --index-url https://token:{SBEROSC_TOKEN}@sberosc.ca.sbrf.ru/repo/pypi/simple --trusted-host sberosc.ca.sbrf.ru")

import sys

sys.path.insert(0, f"{PATH_OMNI_MAIN}/src")
sys.path.insert(0, f"{PATH_OMNI_MAIN}/scripts")
sys.path.insert(0, f"{PATH_ROMASHKA}/src")

import gc
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs

from omnichannel_model.utils import load_pickle
from omnichannel_model.base_classes.base_dataset import HDFSConfig
from omnichannel_model.base_classes.base_calculator import BaseCalculatorClass
from omnichannel_model.omni.models import OmnichannelModel, migrate_old_core_to_new_state_dict
from omnichannel_model.omni.multitask import MultiTaskConfig
from omnichannel_model.omni.configs import OmnichannelModelConfig, OmnichannelModelArtifacts
from omnichannel_model.omni.cycles import to_device
from omnichannel_model.omni.metrics import calc_metrics_dict, flatten_metrics_dict
from omni.utils import create_dataloader, build_dataset, load_evk_model, load_dlt_model

# <<<HELPERS
from scipy.stats import rankdata


def auc_np(score, y):
    """ROC-AUC через ранги (с учётом совпадений); y — булев массив."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap_np(score, y):
    """PR-AUC как average precision (без интерполяции); y — булев массив."""
    if not y.any():
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(prec[ys].sum() / ys.sum())


def resolve_heads(cfg, arg):
    """Имя выхода -> задача. 'auto' берёт все доступные выходы MT-модели."""
    avail = {}
    for t in cfg.tasks:
        if cfg.has_towers(t):
            avail[t] = t
    for t in cfg.general_tasks:
        avail[f"{t}_general"] = t
    for t in cfg.adapter_tasks:
        avail[f"{t}_adj"] = t
    if arg == "auto":
        return avail
    sel = [h.strip() for h in arg.split(",") if h.strip()]
    unknown = [h for h in sel if h not in avail]
    if unknown:
        raise ValueError(f"--heads: неизвестные выходы {unknown}; доступны {list(avail)}")
    return {h: avail[h] for h in sel}


def head_mask(P, T, hi, tidx):
    return (T[:, tidx] >= 0) & ~np.isnan(P[:, hi])


def verdict(lo, hi):
    if np.isnan(lo) or np.isnan(hi):
        return "нет данных"
    return "adj лучше" if lo > 0 else ("adj хуже" if hi < 0 else "различий нет")


def compare_table(P, T, G, names, head_task, task_idx, pairs, ch_name, n_boot=200, max_rows=300000, seed=0):
    """Парный бутстрэп разницы ROC-AUC и PR-AUC между двумя выходами (a - b) по каналам и в целом."""
    rows = []
    rng = np.random.default_rng(seed)
    for a_name, b_name in pairs:
        ia, ib = names.index(a_name), names.index(b_name)
        tidx = task_idx[head_task[a_name]]
        base = head_mask(P, T, ia, tidx) & ~np.isnan(P[:, ib])
        chans = [None] + sorted(np.unique(G[base, 1]).tolist())
        for ch in chans:
            m = base if ch is None else base & (G[:, 1] == ch)
            idx = np.nonzero(m)[0]
            if len(idx) < 200:
                continue
            if len(idx) > max_rows:
                idx = rng.choice(idx, max_rows, replace=False)
            y = T[idx, tidx] == 1
            if y.sum() < 10 or (~y).sum() < 10:
                continue
            a = P[idx, ia].astype(np.float64)
            b = P[idx, ib].astype(np.float64)
            d_auc, d_ap = [], []
            for _ in range(n_boot):
                s = rng.integers(0, len(idx), len(idx))
                ys = y[s]
                if ys.all() or not ys.any():
                    continue
                d_auc.append(auc_np(a[s], ys) - auc_np(b[s], ys))
                d_ap.append(ap_np(a[s], ys) - ap_np(b[s], ys))
            lo_a, hi_a = (np.percentile(d_auc, [2.5, 97.5]) if d_auc else (np.nan, np.nan))
            lo_p, hi_p = (np.percentile(d_ap, [2.5, 97.5]) if d_ap else (np.nan, np.nan))
            auc_a, auc_b, ap_a, ap_b = auc_np(a, y), auc_np(b, y), ap_np(a, y), ap_np(b, y)
            rows.append({
                "пара": f"{a_name} vs {b_name}",
                "канал": "все" if ch is None else ch_name(ch),
                "строк": int(len(idx)), "доля_положит": float(y.mean()),
                "roc_auc_a": auc_a, "roc_auc_b": auc_b, "d_roc_auc": auc_a - auc_b, "d_roc_lo": float(lo_a), "d_roc_hi": float(hi_a),
                "pr_auc_a": ap_a, "pr_auc_b": ap_b, "d_pr_auc": ap_a - ap_b, "d_pr_lo": float(lo_p), "d_pr_hi": float(hi_p),
                "вывод_roc": verdict(lo_a, hi_a), "вывод_pr": verdict(lo_p, hi_p),
            })
    return pd.DataFrame(rows)
# >>>HELPERS


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_weights_into_model(model, ckpt_path):
    obj = _torch_load(ckpt_path)
    sd = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    sd = migrate_old_core_to_new_state_dict(sd)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[CKPT] Загружено из {ckpt_path}. missing={len(miss)}, unexpected={len(unexp)}")
    if miss:
        print(f"[CKPT] Missing keys (первые 10): {list(miss)[:10]}")
    bad = [k for k in miss if k.startswith("mt.")]
    if bad:
        raise RuntimeError(f"[CKPT] в чекпоинте нет весов MT-головы ({len(bad)} ключей mt.*): модель оценивалась бы со случайной инициализацией")
    other = [k for k in miss if k.startswith(("evk_model.", "dlt_model.", "core."))]
    if other:
        print(f"[CKPT] WARNING: не найдены веса подмоделей/ядра ({len(other)}), например {other[:3]}")


@torch.no_grad()
def collect_predictions(model, accelerator, loader, names, total_len, max_batches=None):
    """Один проход по данным: предсказания всех выходов, таргеты и groups. Конец синхронизирован между рангами."""
    device = accelerator.device
    P, T, G = [], [], []
    it = iter(loader)
    exhausted = False
    n_batches = 0
    status = torch.zeros(1, dtype=torch.int32, device=device)
    pbar = tqdm(total=total_len, disable=not accelerator.is_local_main_process)
    while True:
        data = None
        if not exhausted and max_batches and n_batches >= max_batches:
            exhausted = True
        if not exhausted:
            try:
                data = next(it)
                status[0] = 1
            except StopIteration:
                exhausted = True
                status[0] = 0
        else:
            status[0] = 0
        global_status = accelerator.gather(status)
        if data is not None:
            batch = to_device(data, device)
            fmt = batch.pop("data_format")
            with accelerator.autocast():
                preds = model.predict(batch, data_format=fmt)
            missing = [h for h in names if h not in preds]
            if missing:
                raise KeyError(f"model.predict не вернул выходы {missing}; есть {sorted(preds)}")
            P.append(torch.stack([preds[h][:, 0].float() for h in names], dim=1).cpu())
            T.append(batch["target"].cpu().to(torch.int8))
            G.append(batch["groups"].cpu().to(torch.int32))
            n_batches += 1
            pbar.update(1)
        if global_status.max() == 0:
            break
    pbar.close()
    return P, T, G


def main_loop(args):
    torch.cuda.empty_cache()
    gc.collect()
    os.environ.setdefault("NCCL_IB_TIMEOUT", "22")
    os.environ.setdefault("NCCL_IB_RETRY_CNT", "7")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", split_batches=False, kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed)
    is_main = accelerator.is_main_process

    local_root = args.local_root
    artifacts_repo = os.path.join(local_root, "omnichannel_artifacts/omni/")
    artifacts_path = artifacts_repo + "artifacts/"
    weights_dir = artifacts_repo + "weights/"
    metrics_dir = artifacts_repo + "metrics/"
    configs_weights_dir = os.path.join(local_root, "configs_weights/")
    if is_main:
        os.makedirs(metrics_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    ft_dict_name = "omni_ft_dict.pkl"
    preproc_bins_list_name = "omni_preproc_bins.pkl"
    quantile_transformer_name = "omni_preprocessor.pkl"
    cat_encoder_name = "omni_cat_encoder.pkl"
    eval_encoder_name = "omni_eval_encoder.pkl"

    target_columns = [c.strip() for c in args.target_columns.split(",")]
    evaluate_cols = ["eval_prod", "eval_channel"]

    mt_path = args.mt_config if os.path.isabs(args.mt_config) or os.path.exists(args.mt_config) else os.path.join(weights_dir, args.mt_config)
    with open(mt_path) as f:
        mt_cfg = MultiTaskConfig.from_dict(json.load(f))
    heads = resolve_heads(mt_cfg, args.heads)
    names = list(heads)
    max_t = max(mt_cfg.target_cols[t] for t in heads.values())
    if max_t >= len(target_columns):
        raise ValueError(f"--target-columns {target_columns} короче, чем нужно ({max_t + 1} колонок) для выходов {names}")
    task_idx = dict(mt_cfg.target_cols)
    if is_main:
        print(f"[EVAL] выходы: {heads}; target_columns={target_columns}; target_cols={task_idx}")

    ft_dict = load_pickle(artifacts_path + ft_dict_name)
    bins_list = load_pickle(artifacts_path + preproc_bins_list_name)
    cat_feats = [x[0] for x in ft_dict["categorical"]]
    cat_cardinalities = [int(x[1]) for x in ft_dict["categorical"]]
    num_feats = ft_dict["numerical"]
    if len(bins_list) > 0 and isinstance(bins_list[0], np.ndarray):
        bins_list = [torch.tensor(x, dtype=torch.float32) for x in bins_list]
    fin_bins_list = [bins_list[i] for i in range(len(num_feats))]

    decode_maps = None
    eval_encoder_path = artifacts_path + eval_encoder_name
    if os.path.exists(eval_encoder_path):
        eval_encoder = load_pickle(eval_encoder_path)
        decode_maps = {
            "product": {v: k for k, v in eval_encoder["group_name_new"].items()},
            "channel": {v: k for k, v in eval_encoder["channel_id"].items()},
        }
        if is_main:
            print(f"[CONFIG] decode_maps загружены из {eval_encoder_path}")
    else:
        if is_main:
            print(f"[CONFIG] {eval_encoder_path} не найден — метрики будут с сырыми ID продукта/канала")
    ch_name = lambda c: (decode_maps or {}).get("channel", {}).get(int(c), str(int(c)))

    config_omni = OmnichannelModelConfig.load(args.config_path)
    config_omni.model_name = args.mdl_name
    if config_omni.use_emb_flags is None:
        config_omni.use_emb_flags = {"romashka_emb": True, "evk_emb": True, "prod_hier_emb": True, "prod_desc_emb": True}
    if args.use_emb_flags:
        config_omni.use_emb_flags.update(json.loads(args.use_emb_flags))
    if is_main:
        print(f"[CONFIG] use_emb_flags = {config_omni.use_emb_flags}")

    artifacts_pkl_path = artifacts_path + "omni_artifacts.pkl"
    if os.path.exists(artifacts_pkl_path):
        artifact = OmnichannelModelArtifacts.load(artifacts_pkl_path)
    else:
        artifact = OmnichannelModelArtifacts(
            tab_num_bins=fin_bins_list,
            tab_cat_cardinalities=cat_cardinalities,
            deep_kwargs={"num_layers": 4, "growth_rate": 256, "dropout_rate": 0.2, "activation": nn.GELU()},
            cross_kwargs={"rank": 64, "num_experts": 4, "use_nonlinear": False,
                          "moe_gate_type": "sigmoid", "moe_activation": nn.Tanh()},
        )

    evk_model = dlt_model = None
    if config_omni.use_emb_flags.get("evk_emb", False):
        evk_w = os.path.join(configs_weights_dir, args.evk_weights_path) if args.evk_weights_path else None
        evk_model = load_evk_model(config_path=os.path.join(configs_weights_dir, "evk_config.json"), weights_path=evk_w, trainable_patterns=None)
    if config_omni.use_emb_flags.get("romashka_emb", False):
        dlt_w = os.path.join(configs_weights_dir, args.dlt_weights_path) if args.dlt_weights_path else None
        dlt_model = load_dlt_model(config_path=os.path.join(configs_weights_dir, "dlt_config.json"), weights_path=dlt_w, trainable_patterns=None)

    model = OmnichannelModel(config=config_omni, artifacts=artifact, dlt_model=dlt_model, evk_model=evk_model,
                             head_kinds=[], mt_config=mt_cfg)
    ckpt_path = args.chkpt_name if os.path.isabs(args.chkpt_name) else os.path.join(weights_dir, args.chkpt_name)
    if is_main:
        print(f"[EVAL] Loading checkpoint from {ckpt_path}")
    load_weights_into_model(model, ckpt_path)
    model = model.to(accelerator.device)
    model.eval()

    hdfs_cnf = HDFSConfig(
        hdfs_url="hdfs://arnsdpsbx",
        kerb_ticket="/home/datalab/nfs/krb5cc",
        extra_conf={"hadoop.security.authentication": "kerberos",
                    "java.security.krb5.conf": "/home/datalab/nfs/krb5.conf"},
    )
    samp_tbl = {"train": args.train_table, "val": args.val_table, "tst": args.tst_table, "oot": args.oot_table}
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    pairs = [(f"{t}_adj", f"{t}_general") for t in mt_cfg.adapter_tasks if f"{t}_adj" in heads and f"{t}_general" in heads]
    total_len = None
    stamp = datetime.now().date()

    for samp in [s.strip() for s in args.eval_splits.split(",")]:
        if samp not in samp_tbl:
            if is_main:
                print(f"[WARN] Unknown split '{samp}', skipping. Available: {list(samp_tbl)}")
            continue
        table = samp_tbl[samp]
        if is_main:
            print(f"\n--- Evaluating: {samp} ({table}) ---")
        eval_ds = build_dataset(
            config_omni=config_omni, data_path=os.path.join(local_root, table + args.dataset_path),
            num_feats=num_feats, cat_feats=cat_feats, target_columns=target_columns, evaluate_cols=evaluate_cols,
            artifacts_path=artifacts_path, cat_encoder_name=cat_encoder_name, quantile_transformer_name=quantile_transformer_name,
            batch_size=args.batch_size, shuffle=False, seed=args.seed, drop_last=False, unk_emb_prob=0.0,
            file_fraction=args.file_fraction, process_idx=accelerator.process_index, num_processes=accelerator.num_processes,
            hdfs_config=hdfs_cnf, hdfs_data_path=f"/user/team/team_mp_ds_service4bank/hive/{table}", data_format=args.data_format,
        )
        eval_ds.eval()
        eval_ds.total_rows = int(args.dataset_size)
        loader = create_dataloader(eval_ds, num_workers=args.num_workers)
        total_len = int(len(loader) // accelerator.num_processes * 1.1)

        P, T, G = collect_predictions(model, accelerator, loader, names, total_len, args.max_batches)
        n_cols_t = len(target_columns)
        local = {
            "P": torch.cat(P).numpy() if P else np.empty((0, len(names)), np.float32),
            "T": torch.cat(T).numpy() if T else np.empty((0, n_cols_t), np.int8),
            "G": torch.cat(G).numpy() if G else np.empty((0, 2), np.int32),
        }
        del P, T, G, loader
        gc.collect()
        accelerator.wait_for_everyone()
        gathered = [None] * accelerator.num_processes
        dist.all_gather_object(gathered, local)
        del local

        if is_main:
            ok = [g for g in gathered if g["P"].shape[0] > 0]
            if not ok:
                print(f"[RESULT] нет строк для split={samp}")
            else:
                Pn = np.concatenate([g["P"] for g in ok])
                Tn = np.concatenate([g["T"] for g in ok])
                Gn = np.concatenate([g["G"] for g in ok])
                print(f"[EVAL] строк всего: {len(Pn):,}; NaN по выходам: {dict(zip(names, np.isnan(Pn).sum(0).tolist()))}")
                frames = []
                for hi, h in enumerate(names):
                    tidx = task_idx[heads[h]]
                    m = head_mask(Pn, Tn, hi, tidx)
                    if not m.any():
                        print(f"[WARN] {h}: нет валидных строк")
                        continue
                    calc = BaseCalculatorClass(metrics=calc_metrics_dict, modes=modes, device=accelerator.device, decode_maps=decode_maps)
                    calc.update(preds=torch.from_numpy(Pn[m, hi:hi + 1].copy()),
                                targets=torch.from_numpy(Tn[m, tidx:tidx + 1].astype(np.int32)),
                                product_ids=torch.from_numpy(Gn[m, 0].copy()), channel_ids=torch.from_numpy(Gn[m, 1].copy()))
                    res = calc.compute()
                    df = flatten_metrics_dict({h: res}, modes)
                    gen = df[df["slice_type"] == "general"]
                    if len(gen) and float(gen["gini"].iloc[0]) == 0.0 and int(m.sum()) > 1000:
                        print(f"[WARN] {h}: gini=0.0 на {int(m.sum()):,} строк — вероятно, калькулятор проглотил ошибку (формы/NaN)")
                    frames.append(df)
                metrics_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                compare_df = pd.DataFrame()
                if pairs and not args.no_compare:
                    compare_df = compare_table(Pn, Tn, Gn, names, heads, task_idx, pairs, ch_name,
                                               n_boot=args.n_boot, max_rows=args.boot_max_rows, seed=args.seed)
                file_name = f"mt_{stamp}_{samp}_{args.result_name}.xlsx"
                with pd.ExcelWriter(metrics_dir + file_name) as xw:
                    metrics_df.to_excel(xw, sheet_name="metrics", index=False)
                    if len(compare_df):
                        compare_df.to_excel(xw, sheet_name="compare", index=False)
                print(f"[METRICS] Saved: {metrics_dir + file_name}")
                show = ["model", "slice_name", "communications", "avg_cr", "gini", "pr_auc", "logloss", "ece"]
                ch = metrics_df[metrics_df["slice_type"].isin(["general", "channel"])]
                print(ch[[c for c in show if c in ch.columns]].round(4).to_string(index=False))
                if len(compare_df):
                    print("\n[COMPARE] разница выходов a - b (95% бутстрэп-интервал):")
                    print(compare_df[["пара", "канал", "строк", "d_roc_auc", "d_roc_lo", "d_roc_hi", "вывод_roc",
                                      "d_pr_auc", "d_pr_lo", "d_pr_hi", "вывод_pr"]].round(4).to_string(index=False))
        del gathered, eval_ds
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Метрики MT-модели (CR общий/канальный/башни, CTR) за один проход по данным")
    parser.add_argument("--local-root", default="/home/datalab/nfs/", type=str)
    parser.add_argument("--config-path", required=True, type=str, help="omni_config.json (OmnichannelModelConfig)")
    parser.add_argument("--dataset-path", default="", type=str)
    parser.add_argument("--mdl-name", default="OmniModel", type=str)
    parser.add_argument("--chkpt-name", required=True, type=str, help="Файл весов (лучше *_weights.pt) внутри weights_dir или абсолютный путь")
    parser.add_argument("--mt-config", required=True, type=str, help="<mdl_name>_mt_config.json из weights_dir (имя или путь)")
    parser.add_argument("--target-columns", default="trgt_fin,trgt_ctr", type=str, help="Колонки target в том же порядке, что при обучении")
    parser.add_argument("--heads", default="auto", type=str, help="Выходы через запятую (cr_general, cr_adj, ctr, cr) или auto")
    parser.add_argument("--train-table", default="omnichannel_ctr_train", type=str)
    parser.add_argument("--val-table", default="omnichannel_ctr_val", type=str)
    parser.add_argument("--tst-table", default="omnichannel_ctr_tst", type=str)
    parser.add_argument("--oot-table", default="omnichannel_ctr_oot", type=str)
    parser.add_argument("--eval-splits", default="val", type=str, help="train,val,tst,oot через запятую")
    parser.add_argument("--data-format", default="raw", type=str)
    parser.add_argument("--dataset-size", default=1, type=int, help="Оценка размера сплита (для tqdm)")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--file-fraction", default=1.0, type=float, help="Доля файлов (для быстрой оценки на выборке)")
    parser.add_argument("--max-batches", default=0, type=int, help="Ограничить число батчей на ранг (0 = без ограничения)")
    parser.add_argument("--modes", default="general,product,channel,product_channel", type=str)
    parser.add_argument("--n-boot", default=200, type=int, help="Число бутстрэп-выборок для сравнения cr_adj и cr_general")
    parser.add_argument("--boot-max-rows", default=300000, type=int, help="Максимум строк на канал в бутстрэпе")
    parser.add_argument("--no-compare", action="store_true", help="Не считать сравнение cr_adj и cr_general")
    parser.add_argument("--use-emb-flags", default=None, type=str)
    parser.add_argument("--evk-weights-path", default="evk_weights.pt", type=str)
    parser.add_argument("--dlt-weights-path", default="dlt_weights.pt", type=str)
    parser.add_argument("--result-name", default="metrics", type=str)
    args = parser.parse_args()
    print(args)
    main_loop(args)
