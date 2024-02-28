import logging

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict

from marshmallow import INCLUDE, ValidationError
from marshmallow.schema import SchemaMeta

logger: logging.Logger = logging.getLogger(__name__)


class ExtendedEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


class Configurations(ExtendedEnum):
    app_config = "app_config"


class Configuration(ABC):
    def __init__(self, configuration_data: Dict):
        self.configuration_data = configuration_data
        self._validate(configuration_data)

    @abstractmethod
    def _validate(self, configuration_data: Dict) -> bool:
        pass

    def _run_validate(self, check_schema: SchemaMeta, configuration_data: Dict) -> bool:
        validation_errors: Dict = check_schema(unknown=INCLUDE).validate(configuration_data)

        if validation_errors != {}:
            logger.error(f"configuration load error in {self.__class__}: {validation_errors}")
            raise ValidationError(validation_errors)

        return validation_errors == {}
