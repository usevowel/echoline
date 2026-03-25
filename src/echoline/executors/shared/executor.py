from huggingface_hub import ModelCardData
from pydantic import BaseModel

from echoline.api_types import ModelTask
from echoline.model_registry import ModelRegistry


class Executor[ManagerT, RegistryT: ModelRegistry](BaseModel):
    name: str
    model_manager: ManagerT
    model_registry: RegistryT
    task: ModelTask

    model_config = {"arbitrary_types_allowed": True}

    def can_handle_model(self, model_id: str, model_card_data: ModelCardData) -> bool:
        return self.model_registry.hf_model_filter.passes_filter(model_id, model_card_data)
