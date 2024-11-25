from typing import Any
import os
import torch
from lightning import Trainer
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import os
from src.lightning_data import DataModule
from src.lightning_model import LightningModel
# from src.utils.logger import WandbSaveConfigCallback

class ReWriteRootDirCli(LightningCLI):

    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        subcommand = self.subcommand

        # convert local batch_size to global batch_size
        num_nodes = self.config[subcommand]["trainer"]["num_nodes"]
        self.config[subcommand]["tags"]['b'] = str(self.config[subcommand]["tags"]['b']) +f"x{num_nodes}"

        # formulate the root dir
        default_root_dir = self.config[subcommand]["trainer"]["default_root_dir"]
        if default_root_dir is None:
            default_root_dir = os.path.join(os.getcwd(), "workdirs")
        dirname = ""
        for v, k in self.config[subcommand]["tags"].items():
                dirname+=f"_{v}{k}"

        dirname = dirname[1:]
        default_root_dir = os.path.join(default_root_dir, dirname)
        self.config[subcommand]["trainer"]["default_root_dir"] = default_root_dir

        # predict without logger
        if subcommand == "predict":
                self.config[subcommand]["trainer"]["logger"] = None

        # # predict path check
        # if subcommand == "predict":
        #     pred_root = os.path.join(default_root_dir,self.config[subcommand]["model"]["save_dir"])
        #     if os.path.exists(pred_root):
        #         if len(os.listdir(pred_root)) != 0:
        #             raise ValueError(f"Prediction path {pred_root} is not empty")
        # make stupid odps happy
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        class TagsClass:
            def __init__(self, exp:str, b:int|str, s:int|str, e:int,):
                ...
        parser.add_class_arguments(TagsClass, nested_key="tags")
        parser.link_arguments("model.metric.precompute_data_path", "data.test_only_gen_data")

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)
        parser.add_argument("--tables",type=str, default="", help=("make nebu happy" ),)
        parser.add_argument("--torch_hub_dir", type=str, default=None, help=("torch hub dir"),)
        parser.add_argument("--huggingface_cache_dir", type=str, default=None, help=("huggingface hub dir"),)

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        trainer = super().instantiate_trainer(**kwargs)
        return trainer

    def instantiate_classes(self) -> None:
        torch_hub_dir = self._get(self.config, "torch_hub_dir")
        huggingface_cache_dir = self._get(self.config, "huggingface_cache_dir")
        if huggingface_cache_dir is not None:
            os.environ["HUGGINGFACE_HUB_CACHE"] = huggingface_cache_dir
        if torch_hub_dir is not None:
            os.environ["TORCH_HOME"] = torch_hub_dir
            torch.hub.set_dir(torch_hub_dir)
        super().instantiate_classes()

def cli_main():
    # ignore all warnings that could be false positives
    torch.set_float32_matmul_precision('medium')
    cli = ReWriteRootDirCli(LightningModel, DataModule, auto_configure_optimizers=False, save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    cli_main()
