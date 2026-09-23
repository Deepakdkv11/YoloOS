import sys
from pathlib import Path

import hydra
from lightning import Trainer

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel
from yolo.utils.logger import logger
from yolo.utils.logging_utils import setup


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    callbacks, loggers, save_path = setup(cfg)

    trainer = Trainer(
        accelerator=getattr(cfg, "accelerator", "auto"),
        devices=cfg.device,
        max_epochs=getattr(cfg.task, "epoch", None),
        precision="16-mixed",
        callbacks=callbacks,
        sync_batchnorm=True,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        gradient_clip_algorithm="norm",
        deterministic=True,
        enable_progress_bar=not getattr(cfg, "quiet", False),
        default_root_dir=save_path,
    )

    # Optional `+ckpt_path=<file>` to resume an interrupted run. Lightning restores the
    # optimizer, LR schedule, EMA buffers and epoch counter, so training continues where
    # it stopped rather than silently restarting from epoch 0. Essential on Colab, where
    # the runtime is reclaimed mid-run as a matter of course.
    ckpt_path = getattr(cfg, "ckpt_path", None)
    if ckpt_path and not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"ckpt_path={ckpt_path} does not exist")

    if cfg.task.task == "train":
        model = TrainModel(cfg)
        if ckpt_path:
            logger.info(f":arrows_counterclockwise: Resuming training from {ckpt_path}")
        trainer.fit(model, ckpt_path=ckpt_path)
    if cfg.task.task == "validation":
        model = ValidateModel(cfg)
        trainer.validate(model, ckpt_path=ckpt_path)
    if cfg.task.task == "inference":
        model = InferenceModel(cfg)
        trainer.predict(model)


if __name__ == "__main__":
    main()
