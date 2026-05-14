#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
import hydra
from omegaconf import OmegaConf

from utils.utils import (
    extras,
    create_folders,
    task_wrapper,
)
from utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_trainer,
)
from utils.logging_utils import log_hyperparameters
from utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True # This flag defaults to False
torch.backends.cudnn.allow_tf32 = True       # This flag defaults to True


@task_wrapper
def train(cfg):
    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    load_ckpt_path = cfg.get("load_ckpt_path", None)


    # ▼▼▼ 正确加载预训练权重（finetuning模式）▼▼▼
    if load_ckpt_path is not None:
        log.info(f"🔄 Loading pretrained weights for finetuning: {load_ckpt_path}")
        checkpoint = torch.load(load_ckpt_path, map_location="cpu")
        
        # ===== 智能识别checkpoint格式并转换 =====
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # 情况1: 训练保存的完整checkpoint，直接提取
            state_dict = checkpoint["state_dict"]
            log.info("检测到训练checkpoint格式")
        elif isinstance(checkpoint, dict) and not any(k.startswith("model.") for k in checkpoint.keys()):
            # 情况2: 官方发布的纯架构checkpoint，需要添加前缀
            state_dict = {}
            for k, v in checkpoint.items():
                state_dict[f"model.{k}"] = v  # 添加model.前缀
                
            # 初始化EMA权重（官方ckpt没有EMA）
            log.info("检测到官方推理checkpoint，已自动转换键名")
            log.info("⚠️  注意：官方checkpoint不含EMA权重，将用模型权重初始化EMA")
        else:
            # 情况3: 未知格式，尝试直接加载
            state_dict = checkpoint
            log.warning("未知checkpoint格式，尝试直接加载")
        
        # 加载state_dict（strict=False允许部分不匹配）
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        # ===== 初始化EMA（如果缺失）=====
        # 官方ckpt会导致model_ema全部缺失，需要手动初始化
        if any("model_ema." in k for k in missing_keys):
            log.info("初始化EMA权重...")
            model.model_ema.update_parameters(model.model)  # 用模型权重初始化EMA
        
        # ===== 详细报告（保持原有逻辑）=====
        model_missing = [k for k in missing_keys if k.startswith("model.") and not k.startswith("model_ema")]
        ema_missing = [k for k in missing_keys if k.startswith("model_ema.")]
        other_missing = [k for k in missing_keys if not (k.startswith("model.") or k.startswith("model_ema."))]
        
        log.info(f"✅ Loaded checkpoint:")
        log.info(f"   - Model weights missing: {len(model_missing)} keys")
        log.info(f"   - EMA weights missing: {len(ema_missing)} keys")
        log.info(f"   - Other missing: {len(other_missing)} (optimizer states, etc.)")
        if unexpected_keys:
            log.info(f"   - Unexpected keys: {len(unexpected_keys)}")
        
        # 重置ESM（确保在eval模式且冻结）
        model.reset_esm(cfg.model.esm_model, freeze=True)
        
        # 重置微调相关配置
        model.lddt_weight_schedule = cfg.model.get("lddt_weight_schedule", False)
        model.plddt_training = cfg.model.get("plddt_training", False)
        
        log.info(f"📌 Finetuning mode: ESM2 frozen, training only folding network")
    # ▲▲▲ 加载结束 ▲▲▲


    # if load_ckpt_path is not None:
    #     # load existing ckpt
    #     log.info(f"Resuming from checkpoint <{cfg.load_ckpt_path}>...")
    #     model.strict_loading = False

    #     # manually reset these variables in case of fine-tuning
    #     model.lddt_weight_schedule = cfg.model.get("lddt_weight_schedule", False)
    #     model.plddt_training = cfg.model.get("plddt_training", False)

    #     # reset ESM model to avoid issues in loading FSDP checkpoint
    #     model.reset_esm(cfg.model.esm_model)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    OmegaConf.set_struct(cfg.logger, True)
    loggers = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = instantiate_trainer(
        cfg.trainer, callbacks=callbacks, logger=loggers, plugins=None
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    if log:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting training!")
    # trainer.fit(
    #     model=model,
    #     datamodule=datamodule,
    #     ckpt_path=load_ckpt_path,
    # )
    # ✅ 关键：不传 ckpt_path，避免 Lightning 自动恢复
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=None,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="base_train.yaml")
def submit_run(cfg):
    OmegaConf.resolve(cfg)
    extras(cfg)
    create_folders(cfg)
    train(cfg)
    return


if __name__ == "__main__":
    submit_run()
