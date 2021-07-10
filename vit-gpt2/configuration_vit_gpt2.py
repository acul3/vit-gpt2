import copy

from transformers import GPT2Config, ViTConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ViTGPT2Config(PretrainedConfig):

    model_type = "vit-gpt2"
    is_composition = True

    def __init__(self, vit_config_dict, gpt2_config_dict, **kwargs):
        super().__init__(**kwargs)

        if gpt2_config_dict is None:
            raise ValueError("`gpt2_config_dict` can not be `None`.")

        if vit_config_dict is None:
            raise ValueError("`vit_config_dict` can not be `None`.")

        self.gpt2_config = GPT2Config(**gpt2_config_dict)

        self.vit_config = ViTConfig(**vit_config_dict)

    @classmethod
    def from_vit_gpt2_configs(
        cls, vit_config: PretrainedConfig, gpt2_config: PretrainedConfig, **kwargs
    ):
        return cls(
            vit_config_dict=vit_config.to_dict(),
            gpt2_config_dict=gpt2_config.to_dict(),
            **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["vit_config"] = self.vit_config.to_dict()
        output["gpt2_config"] = self.gpt2_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output