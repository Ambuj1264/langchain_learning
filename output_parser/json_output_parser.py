from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import sys
import traceback

load_dotenv()

# NOTE: you saw this warning:
#   Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.
# Options:
#  - Use Python 3.11 or 3.12 (recommended if you want to avoid all compatibility issues).
#  - Or upgrade langchain_core / pydantic packages to versions that work with Python 3.14.
# I'm not forcing any change here — just informing you.

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Parser: many langchain versions return plain dicts from parser.parse().
parser = JsonOutputParser(return_id=True)

# IMPORTANT: include the format instructions inside the template body and reference
# the partial variable {format_instructions} in the template text.
template = PromptTemplate(
    template="""
You are given a piece of text. Produce a short JSON summary following the JSON schema below
(including only the fields required by the schema). Use the JSON format instructions so the
output is parseable as JSON.

Format instructions:
{format_instructions}

Text to summarize:
{text}
""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Try to use the original operator-based composition if it's available in your langchain.
# If that fails, fall back to rendering the prompt, invoking the model, and parsing the output.
user_text = "What is the capital of France?"

try:
    # Many langchain versions support operator composition: template | model | parser
    chain = template | model | parser
    # `invoke` on the composed chain may return a dict, model object, or pydantic model depending on versions.
    result = chain.invoke({"text": user_text})
    # Defensive printing depending on the returned type:
    if result is None:
        print("Result is None")
    elif hasattr(result, "model_dump_json"):
        # pydantic v2 model-like object
        print(result.model_dump_json(indent=2))
    else:
        # For plain dicts (the error you hit), print JSON via json.dumps()
        if isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            # other possible types: string, list, or custom object
            try:
                # try to convert to JSON if possible
                print(json.dumps(result, indent=2))
            except Exception:
                # fallback to plain print with type info
                print("=== chain.invoke returned value ===")
                print("type:", type(result))
                print(result)
except Exception as e:
    # If operator composition failed (common across langchain versions), fall back.
    print("Operator-composition path failed with exception; falling back to manual flow.")
    traceback.print_exc(file=sys.stdout)

    try:
        # Correct usage of format_prompt: use keyword args (don't pass a dict positionally).
        # format_prompt returns a PromptValue; .to_string() gives the rendered string.
        prompt_value = template.format_prompt(text=user_text)
        prompt_str = prompt_value.to_string()

        # Print the prompt for debugging (optional)
        print("---- Rendered prompt ----")
        print(prompt_str)
        print("---- End prompt ----")

        # Invoke the model. Different wrappers return different shapes; print the returned object.
        llm_response = model.invoke(prompt_str)

        print("---- Raw model.invoke return (repr) ----")
        print(repr(llm_response))
        print("---- End raw return ----")

        # Try to normalize raw_text from the model response:
        raw_text = None

        # Common possibilities:
        # - llm_response is a dict with 'content' or 'text'
        # - llm_response is an object with .content attribute (string or list)
        # - llm_response itself is a string
        if isinstance(llm_response, str):
            raw_text = llm_response
        elif isinstance(llm_response, dict):
            # check common keys
            if "content" in llm_response:
                raw_text = llm_response["content"]
            elif "text" in llm_response:
                raw_text = llm_response["text"]
            else:
                # fallback to json dump
                raw_text = json.dumps(llm_response)
        else:
            # object; try known attributes:
            raw_text = getattr(llm_response, "content", None) or getattr(llm_response, "text", None)

        # If content is a list, coerce to string
        if isinstance(raw_text, list):
            # join with newlines or pick first element
            raw_text = "\n".join(str(x) for x in raw_text)

        # If still not a string, coerce to str()
        if raw_text is None:
            raw_text = str(llm_response)

        print("---- Normalized raw_text (what we'll parse) ----")
        print(raw_text)
        print("---- End raw_text ----")

        # Parse using JsonOutputParser
        parsed = parser.parse(raw_text)

        # parser.parse may return a dict or a pydantic model depending on version.
        if hasattr(parsed, "model_dump_json"):
            print(parsed.model_dump_json(indent=2))
        elif isinstance(parsed, dict):
            print(json.dumps(parsed, indent=2))
        else:
            # best effort
            try:
                print(json.dumps(parsed, indent=2))
            except Exception:
                print("Parsed result (fallback print):", parsed)

    except TypeError as te:
        # This catches the "format_prompt() takes 1 positional argument but 2 were given" error,
        # which happens if we called format_prompt(...) with a positional dict instead of using keyword args.
        print("TypeError during fallback flow. Did you pass arguments positionally to format_prompt?")
        traceback.print_exc(file=sys.stdout)
    except Exception:
        print("Unexpected error in fallback flow:")
        traceback.print_exc(file=sys.stdout)