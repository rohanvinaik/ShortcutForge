"""
Build action_catalog.json from Cassinelli action data.

This script accepts either:
- cleaned records (flat dicts with identifier/name/parameters/etc), or
- raw WordPress API records (with `acf` payloads).
"""

import argparse
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CLEAN = BASE_DIR / "references" / "cassinelli_actions_clean.json"
DEFAULT_INPUT_RAW = BASE_DIR / "references" / "cassinelli_actions_raw.json"
DEFAULT_OUTPUT = BASE_DIR / "references" / "action_catalog.json"


def _default_input_path() -> Path:
    if DEFAULT_INPUT_CLEAN.exists():
        return DEFAULT_INPUT_CLEAN
    return DEFAULT_INPUT_RAW


# Clean HTML from parameter strings
def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


# Determine if action produces output based on result field
def produces_output(result: str) -> bool:
    if not result:
        return False
    r = result.lower().strip()
    if r in ("-", "", "does not produce result", "does not produce output"):
        return False
    return True


# Determine if action takes input
def takes_input(input_str: str) -> bool:
    if not input_str:
        return False
    i = input_str.lower().strip()
    if i in ("-", "", "does not take input", "does not accept input"):
        return False
    return True


# Determine if action passes through
def is_passthrough(input_str: str) -> bool:
    if not input_str:
        return False
    return "passes input through" in input_str.lower()


# Categorize by identifier prefix
def get_namespace(identifier: str) -> str:
    if identifier.startswith("is.workflow.actions."):
        return "core"
    if identifier.startswith("com.apple."):
        parts = identifier.split(".")
        return ".".join(parts[:3])
    return "other"


# Map common actions to their actions.make() usage.
# All catalog actions work via actions.make("name", **params). These examples
# document frequently-used actions with their key WF parameter names.
COMPILER_METHODS = {
    "is.workflow.actions.comment": 'actions.make("comment", WFCommentActionText="...")',
    "is.workflow.actions.gettext": 'actions.make("gettext", WFTextActionText=handle)',
    "is.workflow.actions.number": 'actions.make("number", WFNumberActionNumber=42)',
    "is.workflow.actions.url": 'actions.make("url", WFURLActionURL="https://...")',
    "is.workflow.actions.downloadurl": 'actions.make("downloadurl", WFHTTPMethod="GET", WFHTTPHeaders=...)',
    "is.workflow.actions.detect.dictionary": 'actions.make("detect.dictionary")',
    "is.workflow.actions.setvariable": 'actions.make("setvariable", WFVariableName="Name", WFInput=handle)',
    "is.workflow.actions.getvariable": 'actions.make("getvariable", WFVariable=ref_variable("Name"))',
    "is.workflow.actions.getvalueforkey": 'actions.make("getvalueforkey", WFDictionaryKey="key", WFInput=handle)',
    "is.workflow.actions.notification": 'actions.make("notification", WFNotificationActionTitle="Title")',
    "is.workflow.actions.showresult": 'actions.make("showresult", Text=handle)',
    "is.workflow.actions.alert": 'actions.make("alert", WFAlertActionTitle="Title", WFAlertActionMessage="Msg")',
    "is.workflow.actions.ask": 'actions.make("ask", WFAskActionPrompt="Question?")',
    "is.workflow.actions.openurl": 'actions.make("openurl")',
    "is.workflow.actions.health.quantity.log": 'actions.make("health.quantity.log", WFQuantitySampleType="...")',
    "is.workflow.actions.delay": 'actions.make("delay", WFDelayTime=5)',
    "is.workflow.actions.vibrate": 'actions.make("vibrate")',
    "is.workflow.actions.setbrightness": 'actions.make("setbrightness", WFBrightness=0.5)',
    "is.workflow.actions.nothing": 'actions.make("nothing")',
    "is.workflow.actions.exit": 'actions.make("exit")',
    "is.workflow.actions.output": 'actions.make("output", WFOutput=handle)',
    "is.workflow.actions.runworkflow": 'actions.make("runworkflow", WFWorkflowName="Name")',
    "is.workflow.actions.getclipboard": 'actions.make("getclipboard")',
    "is.workflow.actions.setclipboard": 'actions.make("setclipboard", WFInput=handle)',
    "is.workflow.actions.math": 'actions.make("math", WFMathOperand=10, WFMathOperation="+")',
    "is.workflow.actions.count": 'actions.make("count", WFCountType="Items")',
    "is.workflow.actions.list": 'actions.make("list", WFItems=actions.build_list([...]))',
    "is.workflow.actions.choosefromlist": 'actions.make("choosefromlist", WFChooseFromListActionPrompt="Pick")',
    "is.workflow.actions.format.date": 'actions.make("format.date", WFDateFormatString="yyyy-MM-dd")',
    "is.workflow.actions.dictionary": 'actions.make("dictionary", WFItems=actions.build_dict_items({...}))',
    "is.workflow.actions.base64encode": 'actions.make("base64encode", WFEncodeMode="Encode")',
    "is.workflow.actions.urlencode": 'actions.make("urlencode", WFEncodeMode="Encode")',
    "is.workflow.actions.getitemfromlist": 'actions.make("getitemfromlist", WFItemSpecifier="First Item")',
    "is.workflow.actions.openapp": 'actions.make("openapp", WFAppIdentifier="com.apple.mobilesafari")',
    "is.workflow.actions.speaktext": 'actions.make("speaktext", WFSpeakTextRate=0.5)',
    "is.workflow.actions.choosefrommenu": 'CONTROL_FLOW: s.menu_block("Prompt", ["A", "B"])',
    "is.workflow.actions.conditional": "CONTROL_FLOW: s.if_block(handle) / s.if_else_block(handle)",
    "is.workflow.actions.repeat.count": "CONTROL_FLOW: s.repeat_block(count)",
    "is.workflow.actions.repeat.each": "CONTROL_FLOW: s.repeat_each_block(handle)",
}


def _normalize_record(raw: dict) -> dict:
    """Normalize both raw API and clean records into a common shape."""
    if "identifier" in raw and "acf" not in raw:
        return raw

    acf = raw.get("acf", {}) or {}
    title_obj = raw.get("title", {})
    title = ""
    if isinstance(title_obj, dict):
        title = title_obj.get("rendered", "")

    return {
        "identifier": acf.get("identifier", ""),
        "name": acf.get("name", "") or title,
        "description": acf.get("description", ""),
        "parameters": acf.get("parameters", ""),
        "input": acf.get("input", ""),
        "result": acf.get("result", ""),
        "score": acf.get("score", ""),
        "works_well_with_text": acf.get("works_well_with_text", ""),
        "related_shortcuts_text": acf.get("related_shortcuts_text", ""),
        "apps_text": acf.get("apps_text", ""),
    }


def build_catalog(raw_actions: list[dict]) -> dict:
    # Build the catalog
    catalog = {
        "_meta": {
            "description": (
                "Comprehensive Apple Shortcuts action catalog sourced from "
                "Matthew Cassinelli's directory (matthewcassinelli.com)."
            ),
            "source": "matthewcassinelli.com WordPress API",
            "version": "1.0.0",
            "total_actions": 0,
            "actions_with_compiler_method": 0,
            "actions_requiring_make": 0,
        },
        "namespaces": {},
        "actions": {},
    }

    # Process each action
    seen_identifiers = set()
    for raw in raw_actions:
        normalized = _normalize_record(raw)
        ident = normalized.get("identifier", "").strip()
        if not ident or ident in seen_identifiers:
            continue
        seen_identifiers.add(ident)

        namespace = get_namespace(ident)
        name = normalized.get("name", normalized.get("title", "")).strip()
        desc = clean_html(normalized.get("description", ""))
        params = clean_html(normalized.get("parameters", ""))
        input_str = clean_html(normalized.get("input", ""))
        result_str = clean_html(normalized.get("result", ""))
        score = normalized.get("score", "")
        works_with = normalized.get("works_well_with_text", "")
        apps = normalized.get("apps_text", "")

        # Build slug from identifier
        if ident.startswith("is.workflow.actions."):
            slug = ident.replace("is.workflow.actions.", "")
        else:
            slug = ident

        entry = {
            "identifier": ident,
            "name": name,
            "description": desc,
            "parameters_description": params,
            "input": input_str,
            "result": result_str,
            "produces_output": produces_output(result_str),
            "takes_input": takes_input(input_str),
            "passthrough": is_passthrough(input_str),
            "namespace": namespace,
            "score": score,
        }

        if works_with:
            entry["works_well_with"] = works_with
        if apps:
            entry["app_identifier"] = apps

        # Compiler status
        if ident in COMPILER_METHODS:
            entry["compiler_method"] = COMPILER_METHODS[ident]
            entry["compiler_status"] = "implemented"
        else:
            entry["compiler_method"] = f'actions.make("{ident}", ...)'
            entry["compiler_status"] = "use_make"

        catalog["actions"][slug] = entry

        if namespace not in catalog["namespaces"]:
            catalog["namespaces"][namespace] = {"count": 0, "actions": []}
        catalog["namespaces"][namespace]["count"] += 1
        catalog["namespaces"][namespace]["actions"].append(slug)

    # Update meta
    catalog["_meta"]["total_actions"] = len(catalog["actions"])
    catalog["_meta"]["actions_with_compiler_method"] = sum(
        1 for a in catalog["actions"].values() if a["compiler_status"] == "implemented"
    )
    catalog["_meta"]["actions_requiring_make"] = sum(
        1 for a in catalog["actions"].values() if a["compiler_status"] == "use_make"
    )

    # Sort actions within each namespace
    for ns in catalog["namespaces"].values():
        ns["actions"].sort()

    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build action_catalog.json from Cassinelli data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Input JSON (clean or raw Cassinelli action export)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output action_catalog.json path",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. Run scrape_cassinelli.py first."
        )

    with args.input.open() as f:
        raw_actions = json.load(f)
    if not isinstance(raw_actions, list):
        raise ValueError("Input JSON must be a list of action records")

    catalog = build_catalog(raw_actions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"Catalog written to {args.output}")
    print(f"Total actions: {catalog['_meta']['total_actions']}")
    print(f"With compiler methods: {catalog['_meta']['actions_with_compiler_method']}")
    print(
        f"Requiring generic actions.make(): {catalog['_meta']['actions_requiring_make']}"
    )
    print("\nNamespaces:")
    for ns, data in sorted(catalog["namespaces"].items()):
        print(f"  {ns}: {data['count']} actions")


if __name__ == "__main__":
    main()
