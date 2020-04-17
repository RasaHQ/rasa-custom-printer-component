import typing
from typing import Any, Optional, Text, Dict, List, Type

import click
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.tokenizers.tokenizer import Token

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


def _is_list_tokens(v):
    if isinstance(v, List):
        if len(v) > 0:
            if isinstance(v[0], Token):
                return True
    return False


class Debugger(Component):
    """A new component"""

    # Which components are required by this component.
    # Listed components should appear before the component itself in the pipeline.
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline."""

        return []

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {"prefix": "logger", "color": None}

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component.

        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.train`
        of components previous to this one."""
        pass

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.process`
        of components previous to this one."""
        # print(f"{self.prefix}")
        # print(f"text: {message.text}")
        if len(self.component_config['prefix']) > 0:
            click.echo(click.style(self.component_config['prefix'], fg=self.component_config['color']))
        for k, v in message.data.items():
            if _is_list_tokens(v):
                print(f"{k}: {[t.text for t in v]}")
            else:
                print(f"{k}: {v.__repr__()}")

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""
        pass

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file."""

        if cached_component:
            return cached_component
        else:
            return cls(meta)
