import sys
from pathlib import Path

import pytest
import hydra
from omegaconf import DictConfig

import util

sys.path.append(str(Path(__file__).parent.parent))


def train_lightning(backend_config, precision, attn_dropout):
    conf = DictConfig({})
    conf.num_workers = 1
    conf.data_root = Path(util.get_data_paths()["processed_dir"]) / "processed"
    conf.epochs = 1

    conf.dataset = [
        {
            "selection": [
                {
                    "dandiset": "mc_maze_small",
                    "session": "jenkins_20090928_maze",
                }
            ],
            "config": {
                "multitask_readout": [
                    {
                        "decoder_id": "ARMVELOCITY2D",
                        "weight": 2.0,
                        "subtask_key": None,
                        "metrics": [
                            {
                                "metric": "r2",
                                "task": "REACHING",
                            },
                        ],
                    }
                ],
            },
        }
    ]
    conf.seed = 42
    conf.train_transforms = []
    conf.batch_size = 64
    conf.model = DictConfig(
        {
            "_target_": "kirby.models.POYOPlus",
            "dim": 64,
            "dim_head": 64,
            "num_latents": 16,
            "depth": 1,
            "cross_heads": 2,
            "self_heads": 8,
            "ffn_dropout": 0.2,
            "lin_dropout": 0.4,
            "atn_dropout": attn_dropout,
        }
    )

    # Make sure we run every iteration so we try out validation.
    conf.eval_epochs = 1
    conf.base_lr = 1.5625e-5
    conf.pct_start = 0.5
    conf.num_workers = 4
    conf.log_dir = "./logs"
    conf.name = "poyo_single_session"
    conf.ckpt_path = None
    conf.weight_decay = 1e-4
    conf.backend_config = backend_config

    conf.epochs = 2
    conf.steps = 0
    conf.freeze_perceiver_until_epoch = 0
    conf.finetune = False
    conf.precision = precision
    conf.nodes = 1
    conf.gpus = 1

    from train import run_training  # noqa: E402

    run_training(conf)


def test_train_poyo_plus_cpu():
    train_lightning("cpu", 32, 0.2)


def test_train_poyo_plus_gpu_fp32():
    train_lightning("gpu_fp32", 32, 0.2)


def test_train_poyo_plus_gpu_fp32_var():
    train_lightning("gpu_fp32_var", 32, 0)

    # gpu_fp32_var shouldn't allow atn_dropout = 0 in the cross attention
    with pytest.raises((hydra.errors.InstantiationException, ValueError)):
        train_lightning("gpu_fp32_var", 32, 0.2)


# this test will only work with GPUs supported by flashattn
def test_train_poyo_plus_gpu_fp16():
    try:
        train_lightning("gpu_fp16", 16, 0.2)
    except ValueError as e:
        if "requires device with capability" in str(e):
            pytest.xfail(
                "This test is expected to fail due to GPU capability requirements."
            )
        else:
            pytest.fail(f"Unexpected ValueError: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")
