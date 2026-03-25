import json

from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from pydantic import TypeAdapter

adapter = TypeAdapter(CompletionCreateParamsBase)

print(json.dumps(adapter.json_schema()))

# python src/echoline/types/generate_chat_json_schema.py | datamodel-codegen --input-file-type jsonschema --target-python-version $(cat .python-version) --output-model-type pydantic_v2.BaseModel --use-standard-collections --use-union-operator --enum-field-as-literal all > src/echoline/types/chat.py && ruff check --fix src/echoline/types/chat.py && ruff format src/echoline/types/chat.py
