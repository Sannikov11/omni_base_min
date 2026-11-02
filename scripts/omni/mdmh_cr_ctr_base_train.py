import os

print(os.getcwd())
# ----------------------------------------------------------
# Подтянуть токен и пути из .env (положить в home/datala/nfs/.env) 
# SBEROSC_TOKEN=[TOKEN]
# PATH_OMNI_MAIN=/home/datalab/nfs/Omni/omnichannel_main
# PATH_ROMASHKA=/home/datalab/nfs/Omni/Romashka/camomile_models
# POLARS_MAX_THREADS=1
# RAYON_NUM_THREADS=1
# ----------------------------------------------------------
with open(".env") as f: 
    for line in f: 
        line = line.strip()
        if line and not line.startswith("#") and "=" in line: 
            key, value = line.split("=", 1)
            os.environ[key] = value
            print(f"Key:{key}, Value:{value}")

def _require_env(var_name: str, hint: str) -> str: 
    value = os.environ.get(var_name)
    if not value: 
        raise RuntimeError(
            f"Переменная окружения {var_name} не задана ({hint})"
            f"Задайте ее перд запуском: export {var_name}=..., либо в .env"
        )
    return value

POLARS_MAX_THREADS = _require_env("POLARS_MAX_THREADS", "")
RAYON_NUM_THREADS = _require_env("RAYON_NUM_THREADS", "")

PATH_OMNI_MAIN = _require_env("PATH_OMNI_MAIN", "/home/datalab/nfs/Omni/omnichannel_main")
PATH_ROMASHKA = _require_env("PATH_ROMASHKA", "/home/datalab/nfs/Omni/Romashka/camomile_models")

SBEROSC_TOKEN = _require_env("SBEROSC_TOKEN", "Токен доступа к SberOSC для доступа к PyPI")

os.system(f"pip install polars==1.29.0 --index-url https://token:{SBEROSC_TOKEN}@sberosc.ca.sbrf.ru/repo/pypi/simple --trusted-host sberosc.ca.sbrf.ru")
os.system(f"pip install torchmetrics --index-url https://token:{SBEROSC_TOKEN}@sberosc.ca.sbrf.ru/repo/pypi/simple --trusted-host sberosc.ca.sbrf.ru")

print(f"PATH_OMNI_MAIN:{PATH_OMNI_MAIN}")
print(f"PATH_ROMASHKA:{PATH_ROMASHKA}")

import sys
sys.path.insert(0, f"{PATH_OMNI_MAIN}/src")
sys.path.insert(0, f"{PATH_OMNI_MAIN}/scripts")
sys.path.insert(0, f"{PATH_ROMASHKA}/src")

import gc
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta
from collections import OrderedDict
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs, InitProcessGroupKwargs
import torch.distributed as dist


from omnichannel_model.utils import load_pickle
from omnichannel_model.base_classes.base_dataset import HDFSConfig
from omnichannel_model.omni.models import OmnichannelModel
from omnichannel_model.omni.configs import OmnichannelModelConfig, OmnichannelModelArtifacts
from omnichannel_model.omni.dataset import OmniTrainDataset
from omnichannel_model.omni.utils import build_optimizer, build_scheduler
import omnichannel_model.omni.cycles as cycles
from omni.utils import create_dataloader, build_dataset, resolve_emb_dataset_kwargs, load_evk_model, load_dlt_model

from omnichannel_model.evk.models import CommunicationModelFull, CommunicationModelConfig
from camomile_models.models.dlt.DLTmodel import DLTmodelBase

import omnichannel_model.omni.cycles_mt as cycles_mt
from omnichannel_model.omni.multitask import MultiTaskConfig

from dataclasses import asdict




# ---------------------------------------------------------------------------
# Head-specific training dispatch
# ---------------------------------------------------------------------------

HEAD_TRAIN_FN = {
    "cr":       cycles.train_cr_head_osiris,
    "ctr":      cycles.train_ctr_head_osiris,
    "pairwise": cycles.train_pairwise_head_osiris,
}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main_loop(args):
    torch.cuda.empty_cache()
    gc.collect()

    # Env tweaks for NCCL stability
    os.environ.setdefault("NCCL_IB_TIMEOUT", "22")
    os.environ.setdefault("NCCL_IB_RETRY_CNT", "7")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        split_batches=False,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(args.seed)

    # ----------------------------------------------------------
    # Paths
    # ----------------------------------------------------------
    local_root     = args.local_root
    artifacts_repo = os.path.join(local_root, "omnichannel_artifacts/omni/")
    artifacts_path = artifacts_repo + "artifacts/"
    weights_dir    = artifacts_repo + "weights/"
    metrics_dir    = artifacts_repo + "metrics/"
    configs_weights_dir = os.path.join(local_root, "configs_weights/")

    if accelerator.is_main_process:
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # ----------------------------------------------------------
    # Artifact file names (matching ctr_dpp.py / omni_dpp_refactored-3.py)
    # ----------------------------------------------------------
    ft_dict_name              = "omni_ft_dict.pkl"
    preproc_bins_list_name    = "omni_preproc_bins.pkl"
    quantile_transformer_name = "omni_preprocessor.pkl"
    cat_encoder_name          = "omni_cat_encoder.pkl"
    prod_hier_name            = "prod_hier_emb.pt"
    prod_desc_name            = "prod_desc_emb.pt"

    target_columns = [c.strip() for c in args.target_columns.split(",")]#["trgt_fin"]
    

    # ----------------------------------------------------------
    # Load feature dict & bins
    # ----------------------------------------------------------
    ft_dict      = load_pickle(artifacts_path + ft_dict_name)
    bins_list    = load_pickle(artifacts_path + preproc_bins_list_name)
    cat_feats    = [x[0] for x in ft_dict["categorical"]]
    cat_cardinalities = [int(x[1]) for x in ft_dict["categorical"]]
    num_feats    = ft_dict["numerical"]

    if len(bins_list) > 0 and isinstance(bins_list[0], np.ndarray):
        bins_list = [torch.tensor(x, dtype=torch.float32) for x in bins_list]
    fin_bins_list = [bins_list[i] for i in range(len(num_feats))]

    # ----------------------------------------------------------
    # Model config & artifacts
    # ----------------------------------------------------------
    config_omni = OmnichannelModelConfig.load(args.config_path)
    config_omni.model_name = args.mdl_name
    
    if config_omni.use_emb_flags is None:
            config_omni.use_emb_flags = {
                'romashka_emb': False,
                'evk_emb': True,
                'prod_hier_emb': True, 
                'prod_desc_emb': True
            }
            
    if accelerator.is_main_process:
        print(f"[CONFIG] Loaded from {args.config_path}")
        

    # Override use_emb_flags from CLI if provided
    if args.use_emb_flags:
        override = json.loads(args.use_emb_flags)
        config_omni.use_emb_flags.update(override)
        if accelerator.is_main_process:
            print(f"[CONFIG] use_emb_flags overridden via CLI: {override}")
    else : 
        if accelerator.is_main_process:
            print("[CONFIG] use_emb_flags use default values")
            
    
        
    
    if accelerator.is_main_process:
        print(f"[CONFIG] use_emb_flags = {config_omni.use_emb_flags}")

    artifacts_pkl_path = artifacts_path + "omni_artifacts.pkl"
    if os.path.exists(artifacts_pkl_path):
        artifact = OmnichannelModelArtifacts.load(artifacts_pkl_path)
        if accelerator.is_main_process:
            print("[ARTIFACTS] Loaded from file.")
    else:
        if accelerator.is_main_process:
            print("[ARTIFACTS] Building from scratch.")
        artifact = OmnichannelModelArtifacts(
            tab_num_bins=fin_bins_list,
            tab_cat_cardinalities=cat_cardinalities,
            deep_kwargs={
                "num_layers": 4, "growth_rate": 256,
                "dropout_rate": 0.2, "activation": nn.GELU(),
            },
            cross_kwargs={
                "rank": 64, "num_experts": 4, "use_nonlinear": False,
                "moe_gate_type": "sigmoid", "moe_activation": nn.Tanh(),
            },
        )

    # ----------------------------------------------------------
    # Sub-models (EVK + DLT)
    # ----------------------------------------------------------
    if config_omni.use_emb_flags.get('evk_emb', False):
        evk_weights_path = (
            os.path.join(configs_weights_dir, args.evk_weights_path)
            if args.evk_weights_path else None            # None -> обучение EVK с нуля
        )
        evk_trainable = json.loads(args.evk_trainable_patterns) if args.evk_trainable_patterns else None
        evk_model = load_evk_model(
            config_path=os.path.join(configs_weights_dir, "evk_config.json"),
            weights_path=evk_weights_path,
            trainable_patterns=evk_trainable,
        )
    else:
        evk_model = None
    
    if config_omni.use_emb_flags.get('romashka_emb', False):
        dlt_weights_path = (
            os.path.join(configs_weights_dir, args.dlt_weights_path)
            if args.dlt_weights_path else None             # None -> обучение DLT с нуля
        )
        dlt_trainable = json.loads(args.dlt_trainable_patterns) if args.dlt_trainable_patterns else None
        dlt_model = load_dlt_model(
            config_path=os.path.join(configs_weights_dir, "dlt_config.json"),
            weights_path=dlt_weights_path,
            trainable_patterns=dlt_trainable,
        )
    else:
        dlt_model = None

    # ----------------------------------------------------------
    # Build OmnichannelModel
    # ----------------------------------------------------------
    head_kinds = [k for k in args.head_kinds.split(",") if k]
    mt_config = None
    if args.multitask:
        cps = json.loads(args.channels_per_task)
        ht_mt = json.loads(args.head_targets)
        miss = [c for c in ht_mt.values() if c not in target_columns]
        if set(ht_mt) != set(cps) or miss:
            raise ValueError(f"--head-targets {ht_mt} должны совпадать по задачам с --channels-per-task {cps} и ссылаться на --target-columns {target_columns}")
        mt_dict = json.loads(args.mt_config) if args.mt_config else {}
        mt_dict["tasks"] = list(cps)
        mt_dict["channels_per_task"] = cps
        mt_dict["target_cols"] = {h: target_columns.index(c) for h, c in ht_mt.items()}
        if args.head_weights:
            mt_dict["task_weights"] = json.loads(args.head_weights)
        mt_config = MultiTaskConfig.from_dict(mt_dict)
        head_kinds = []
        if accelerator.is_main_process:
            print(f"[MT] multitask-режим, --head-kinds игнорируется (pairwise не поддержан): {mt_config}")


    head_targets, head_weights = None, None
    
    if args.joint and not args.multitask:
        ht = json.loads(args.head_targets)
        if set(ht) != set(head_kinds):
            raise ValueError(f"--head-targets {sorted(ht)} должны совпадать с --head-kinds {sorted(head_kinds)}")
        if not set(head_kinds) <= {"cr", "ctr"}:
            raise ValueError("--joint поддерживает только pointwise-головы cr и ctr")
        missing = [c for c in ht.values() if c not in target_columns]
        #if missing:
            #raise ValueError(f"Колонки {missing} отсутствуют в --target-columns {target_columns}")
        head_targets = {h: target_columns.index(c) for h, c in ht.items()}
        head_weights = json.loads(args.head_weights) if args.head_weights else {h: 1.0 for h in head_kinds}
        if set(head_weights) != set(head_kinds) or any(float(v) <= 0 for v in head_weights.values()):
            raise ValueError("--head-weights: ключи должны совпадать с головами, значения > 0")


    model = OmnichannelModel(
        config    = config_omni,
        artifacts = artifact,
        dlt_model = dlt_model,
        evk_model = evk_model,
        head_kinds = head_kinds,
        mt_config = mt_config,
    )

    if accelerator.is_main_process:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL] Total params: {total:,}  Trainable: {trainable:,}")

    # ----------------------------------------------------------
    # Optional: freeze backbone, unfreeze heads only
    # ----------------------------------------------------------
    if args.freeze_backbone:
        model.freeze_layers()
        for head_name in head_kinds:
            head = model.heads[head_name]
            layer_names = [f"heads.{head_name}.{n}" for n in head.get_layer_names()]
            model.unfreeze_layers(layer_names)
        if mt_config is not None:
            model.unfreeze_layers(["mt"])
        if accelerator.is_main_process:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            print(f"[FREEZE] Trainable {trainable:,} / {total:,} params")
            
    if args.train_adapter_only:
        if mt_config is None or not mt_config.adapter_tasks:
            raise ValueError("--train-adapter-only требует --multitask и adapter_tasks в --mt-config")
        model.freeze_layers()
        model.unfreeze_layers(["mt.adapters"])
        if accelerator.is_main_process:
            print(f"[ADAPTER-ONLY] обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------
    samp_tbl = {
        "train": args.train_table,
        "val":   args.val_table,
        "tst":   args.tst_table,
        "oot":   args.oot_table,
    }
    batch_size = args.batch_size
    EPOCHS     = args.epochs
    samp_cnt_train = args.rows_cnt   # approximate; used for scheduler steps
    steps      = samp_cnt_train // batch_size

    if accelerator.is_main_process:
        print(f"[CONFIG] KRB5CCNAME = {os.environ.get('KRB5CCNAME')}")

    hdfs_cnf = HDFSConfig(
        hdfs_url   = "hdfs://arnsdpsbx",
        kerb_ticket = "/home/datalab/nfs/krb5cc",
        extra_conf  = {
            "hadoop.security.authentication": "kerberos",
            "java.security.krb5.conf":        "/home/datalab/nfs/krb5.conf",
        },
    )

    train_ds = build_dataset(
        config_omni             = config_omni,
        data_path               = os.path.join(local_root, samp_tbl["train"] + args.dataset_path),
        num_feats               = num_feats,
        cat_feats               = cat_feats,
        target_columns          = target_columns,
        evaluate_cols           = [c.strip() for c in args.evaluate_cols.split(",")] if args.multitask else [],
        artifacts_path          = artifacts_path,
        cat_encoder_name        = cat_encoder_name,
        quantile_transformer_name = quantile_transformer_name,
        batch_size              = batch_size,
        shuffle                 = True,
        seed                    = args.seed,
        drop_last               = True,
        unk_emb_prob            = args.unk_emb_prob,
        file_fraction           = args.file_fraction,
        process_idx             = accelerator.process_index,
        num_processes           = accelerator.num_processes,
        hdfs_config             = hdfs_cnf,
        hdfs_data_path          = f"/user/team/team_mp_ds_service4bank/hive/{samp_tbl['train']}",
        data_format             = args.data_format
    )

    train_ds.train()
    train_ds.total_rows = samp_cnt_train

    # ----------------------------------------------------------
    # Optimizer & scheduler (layerwise LRs from utils.py)
    # ----------------------------------------------------------
    start_epoch = args.start_epoch
    optimizer   = build_optimizer(
        model,
        mt_lrs=json.loads(args.mt_lrs) if mt_config is not None else None,
        mt_wds=json.loads(args.mt_wds) if mt_config is not None else None,
    )
    scheduler   = build_scheduler(optimizer, steps, EPOCHS)

    # ----------------------------------------------------------
    # Resume from checkpoint
    # ----------------------------------------------------------
    if args.resume_from is not None:
        if accelerator.is_main_process:
            print(f"[RESUME] Loading from {args.resume_from}")
        ckpt_meta = model.load_checkpoint(
            path           = args.resume_from,
            strict         = False,
            load_optimizer = not args.no_load_optimizer,
            optimizer      = optimizer,
        )
        _ck = torch.load(args.resume_from, map_location="cpu")
        sched_state = _ck.get("scheduler_state_dict") or (_ck.get("extra_data") or {}).get("scheduler_state_dict")
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)
        if ckpt_meta.get("epoch") is not None:
            start_epoch = ckpt_meta["epoch"] + 1
        if accelerator.is_main_process:
            print(f"[RESUME] Epoch={ckpt_meta.get('epoch')}, ts={ckpt_meta.get('timestamp')}")

    # ----------------------------------------------------------
    # TensorBoard
    # ----------------------------------------------------------
    writer = None
    if accelerator.is_main_process:
        log_name = f"omni_{datetime.now().date()}"
        writer   = SummaryWriter(log_name)
        print(f"[TENSORBOARD] logdir={log_name}")

    # ----------------------------------------------------------
    # Accelerate prepare
    # ----------------------------------------------------------
    optimizer, scheduler, model = accelerator.prepare(optimizer, scheduler, model)

    # ----------------------------------------------------------
    # Training loop — iterates over requested heads
    # ----------------------------------------------------------
    for epoch in range(start_epoch, EPOCHS):
        if accelerator.is_main_process:
            print("=" * 60)
            print(f"[EPOCH {epoch + 1}/{EPOCHS}]")
            print("=" * 60)

        train_ds.set_epoch(epoch)
        train_loader = create_dataloader(
            train_ds,
            num_workers       = int(args.num_workers),
            persistent_workers = int(args.num_workers) > 0,
        )
        model.train()

        epoch_losses = {}
        
        if args.multitask:
            epoch_losses = cycles_mt.train_multitask_osiris(
                model=model, accelerator=accelerator, loader=train_loader,
                optimizer=optimizer, scheduler=scheduler, writer=writer, epoch=epoch,
                log_every=args.log_every, gc_every=args.gc_every,
                channel_names=json.loads(args.channel_names),
                step_offset=epoch * (args.rows_cnt // (batch_size * accelerator.num_processes)),
            )
        elif args.joint:
            epoch_losses = cycles.train_multihead_osiris(
                model=model, accelerator=accelerator, loader=train_loader,
                optimizer=optimizer, scheduler=scheduler, writer=writer, epoch=epoch,
                head_targets=head_targets, head_weights=head_weights,
                log_every=args.log_every, diag_every=args.diag_every,
                step_offset=epoch * (args.rows_cnt // (batch_size * accelerator.num_processes)),
            )
        for head_name in ([] if (args.joint or args.multitask) else head_kinds):
            if head_name not in HEAD_TRAIN_FN:
                print(f"[WARN] Unknown head '{head_name}', skipping.")
                continue

            train_fn = HEAD_TRAIN_FN[head_name]
            train_loss = train_fn(
                model       = model,
                accelerator = accelerator,
                loader      = train_loader,
                optimizer   = optimizer,
                scheduler   = scheduler,
                writer      = writer,
                epoch       = epoch,
            )
            epoch_losses[head_name] = train_loss

        accelerator.wait_for_everyone()

        # ----------------------------------------------------------
        # Checkpoint (main process)
        # ----------------------------------------------------------
        if accelerator.is_main_process:
            for head_name, train_loss in epoch_losses.items():
                writer.add_scalar(f"Train epoch loss/{head_name}", train_loss, epoch + 1)

            unwrapped_model = accelerator.unwrap_model(model)

            # Full checkpoint (model weights + optimizer + scheduler + epoch)
            ckpt_path = os.path.join(
                weights_dir,
                f"{config_omni.model_name}_ep{epoch}.pt",
            )
            unwrapped_model.save_checkpoint(
                path      = ckpt_path,
                optimizer = optimizer,
                epoch     = epoch,
                metadata  = {"train_losses": epoch_losses},
                extra_data = {
                    "scheduler_state_dict": scheduler.state_dict(),
                },
            )
            print(f"[CKPT] Saved: {ckpt_path}")

            # Lightweight weights-only file (for inference / resume)
            weights_path = os.path.join(
                weights_dir,
                f"{config_omni.model_name}_ep{epoch}_weights.pt",
            )
            unwrapped_model.save_weights(weights_path)
            print(f"[WEIGHTS] Saved: {weights_path}")

            # Save config snapshot next to weights
            unwrapped_model.save_config(
                os.path.join(weights_dir, f"{config_omni.model_name}_config.json")
            )
            
            if mt_config is not None:
                with open(os.path.join(weights_dir, f"{config_omni.model_name}_mt_config.json"), "w") as f:
                    json.dump(asdict(mt_config), f, ensure_ascii=False, indent=2)

        del train_loader
        gc.collect()
        torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------
    del train_ds
    gc.collect()
    if writer is not None:
        writer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OmnichannelModel distributed training script"
    )

    # Paths
    parser.add_argument("--local-root",   default="/home/datalab/nfs/", type=str)
    parser.add_argument("--config-path",  required=True,  type=str,
                        help="Path to omni_config.json (OmnichannelModelConfig)")
    parser.add_argument("--dataset-path", default="",     type=str,
                        help="Suffix appended to local_root/train_table path")
    parser.add_argument("--mdl-name",     default="OmniModel", type=str)

    # Table names (Hive / local)
    parser.add_argument("--train-table", default="omnichannel_ctr_train", type=str)
    parser.add_argument("--val-table",   default="omnichannel_ctr_val",   type=str)
    parser.add_argument("--tst-table",   default="omnichannel_ctr_tst",   type=str)
    parser.add_argument("--oot-table",   default="omnichannel_ctr_oot",   type=str)
    
    parser.add_argument("--data-format", default="raw", type=str,
                       help="raw/precomputed for dataset thet declare what kind of embeds we have")
    
    # Training hypers
    parser.add_argument("--rows-cnt",     required=True, type=int,
                    help="Глобальное число строк train (определяет расписание LR)")
    parser.add_argument("--epochs",       default=1,    type=int)
    parser.add_argument("--batch-size",   default=256,  type=int)
    parser.add_argument("--seed",         default=42,   type=int)
    parser.add_argument("--num-workers",  default=2,    type=int)
    parser.add_argument("--file-fraction",default=1.0,  type=float)
    parser.add_argument("--unk-emb-prob", default=0.1,  type=float)

    # Heads to train (comma-separated subset of: cr, pairwise, ctr)
    parser.add_argument("--target-columns",   default="trgt_fin", type=str,
                        help="Comma-separated target names to train: target_cr, target_ctr, etc.")
    
    parser.add_argument("--head-kinds",   default="cr,pairwise,ctr", type=str,
                        help="Comma-separated head names to train: cr, pairwise, ctr")
    
    parser.add_argument("--channel-split", action="store_true",
                        help="True if there are 2 or more channels")
    

    # Embeddings override
    parser.add_argument(
        "--use-emb-flags", default=None, type=str,
        help='JSON string to override use_emb_flags, e.g. \'{"romashka_emb": false}\''
    )

    # EVK / DLT: веса и режим обучения
    parser.add_argument("--evk-weights-path", default=None, type=str,
                        help="Имя файла весов EVK внутри configs_weights_dir. "
                            "Не указывайте — обучение EVK с нуля")
    parser.add_argument("--dlt-weights-path", default=None, type=str,
                        help="Имя файла весов DLT внутри configs_weights_dir. "
                            "Не указывайте — обучение DLT с нуля")
    parser.add_argument("--evk-trainable-patterns", default=None, type=str,
                        help='JSON-список подстрок обучаемых параметров EVK, '
                            'например \'["Encoder.2", "Encoder.3"]\'. '
                            'Не указан — все параметры заморожены (если веса загружены)')
    parser.add_argument("--dlt-trainable-patterns", default=None, type=str,
                        help='JSON-список подстрок обучаемых параметров DLT, '
                            'например \'["Model.decoder.4", "Model.decoder.5"]\'')
    # Resume / fine-tuning
    parser.add_argument("--resume-from",    default=None, type=str,
                        help="Path to .pt checkpoint to resume from")
    parser.add_argument("--start-epoch",    default=0,    type=int,
                        help="Starting epoch index (0-based); ignored if --resume-from is set")
    parser.add_argument("--freeze-backbone",action="store_true",
                        help="Freeze backbone, train prediction heads only")
    
    parser.add_argument("--joint", action="store_true",
                        help="Совместное обучение голов за один проход")
    parser.add_argument("--head-targets", default='{"cr":"trgt_cr","ctr":"trgt_ctr"}', type=str)
    parser.add_argument("--head-weights", default=None, type=str,
                        help='JSON, например {"cr":1.0,"ctr":1.0}; по умолчанию 1.0 для всех')
    parser.add_argument("--log-every", default=10, type=int)
    parser.add_argument("--diag-every", default=100, type=int)

    
    parser.add_argument("--multitask", action="store_true",
                    help="Единая multi-task схема (MultiTaskHead): выходы = задача x канал")
    parser.add_argument("--channels-per-task", default='{"cr":4,"ctr":1}', type=str)
    parser.add_argument("--mt-config", default=None, type=str,
                        help="JSON с полями MultiTaskConfig (эксперты, башни, pos_weight, channel_ids ...)")
    parser.add_argument("--evaluate-cols", default="eval_prod,eval_channel", type=str,
                        help="Колонки groups; индекс 1 = канал (MultiTaskConfig.channel_col). Должны быть int в parquet")
    parser.add_argument("--channel-names", default='{"cr":["mass","phygital","premier","sbol"]}', type=str)
    parser.add_argument("--mt-lrs", default='{"experts":1e-3,"gates":5e-4,"towers":1e-3,"adapter":1e-3}', type=str)
    parser.add_argument("--mt-wds", default='{"experts":1e-2,"gates":1e-3,"towers":1e-2,"adapter":1e-2}', type=str)

    parser.add_argument("--gc-every", default=0, type=int,
                        help="Каждые N шагов считать косинус градиентов задач на shared-экспертах (0=выкл, +1 backward по задаче)")
    parser.add_argument("--no-load-optimizer", action="store_true",
                        help="При --resume-from не грузить состояние оптимизатора (нужно при смене числа param-групп)")
    parser.add_argument("--no-find-unused", action="store_true",
                        help="DDP find_unused_parameters=False (быстрее; проверьте на --multitask)")
    
    parser.add_argument("--train-adapter-only", action="store_true",
                    help="Заморозить всё, кроме mt.adapters (канальная поправка поверх готовой базы; веса базы грузить через --resume-from)")


    args = parser.parse_args()
    print(args)
    main_loop(args)
