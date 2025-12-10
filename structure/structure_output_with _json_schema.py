from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json

load_dotenv()

model = ChatOpenAI()

# JSON Schema for Review
review_json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key themes of the review"
        },
        "summary": {
            "type": "string",
            "description": "Summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "Sentiment of the review"
        },
        "pros": {
            "type": "array",
            "items": {"type": "string"},
            "description": "best value phone"
        },
        "cons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write all the cons of the review"
        },
        "rating": {
            "type": "integer",
            "description": "Write the rating of the review"
        },
        "name": {
            "type": "string",
            "description": "Write the name of the review"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

# Structured output using JSON Schema
structureModel = model.with_structured_output(schema=review_json_schema)

result = structureModel.invoke(
    "I recently upgraded to the Samsung Galaxy S23 Ultra, "
    "and I am extremely satisfied with it. It is the best phone I have ever used. "
    "I have been using it for the past 2 months. Review by Ambuj Singh."
)

print(json.dumps(result, indent=2))
