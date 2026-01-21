#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from urllib import request, error

DEFAULT_RULES_FILE = "rule_description.txt"
DEFAULT_STATE_FILE = ".rules_state.json"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CODE_FILES = [
    "internal/query/querybuilder.go",
]


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_rules(text):
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return blocks


def hash_rule(rule_text):
    return hashlib.sha256(rule_text.encode("utf-8")).hexdigest()


def load_state(path):
    if not os.path.exists(path):
        return {"rules": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"rules": {}}


def save_state(path, state):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=True)


def get_new_rules(rules, state):
    new_rules = []
    existing = state.get("rules", {})
    for r in rules:
        h = hash_rule(r)
        if h not in existing:
            new_rules.append((h, r))
    return new_rules


def collect_files(paths):
    files = []
    for p in paths:
        if os.path.exists(p) and os.path.isfile(p):
            files.append(p)
    return files


def format_files_for_prompt(files):
    parts = []
    for p in files:
        parts.append(f"File: {p}\n")
        parts.append(read_text(p))
        parts.append("\n\n")
    return "".join(parts)


def openai_chat(api_key, model, system_prompt, user_prompt, temperature=0.0, api_base=None):
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    url = (api_base or "https://api.openai.com") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = request.Request(url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as e:
        raise RuntimeError(f"OpenAI API error: {e.read().decode('utf-8')}")
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")
    parsed = json.loads(body)
    content = parsed["choices"][0]["message"]["content"]
    return content


def apply_diff(diff_text):
    if not diff_text.strip():
        raise RuntimeError("Empty diff returned by LLM.")
    patch_path = ".llm_patch.diff"
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(diff_text)
    cmd = ["git", "apply", patch_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"git apply failed: {res.stderr.strip()}")
    os.remove(patch_path)


def extract_changed_go_files(diff_text):
    files = set()
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/") :].strip()
            if path.endswith(".go"):
                files.add(path)
    return sorted(files)


def gofmt_files(files):
    if not files:
        return
    cmd = ["gofmt", "-w"] + files
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"gofmt failed: {res.stderr.strip()}")


def run_cmd(cmd, cwd=None):
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def build_and_test(repo_root, scenario_args):
    run_cmd(["go", "build", "-o", "det-tool"], cwd=repo_root)
    run_cmd(["go", "build", "-o", os.path.join("cmd", "scenarios", "run-scenarios"), "./cmd/scenarios"], cwd=repo_root)
    cmd = [os.path.join("cmd", "scenarios", "run-scenarios")]
    if scenario_args:
        cmd += scenario_args
    run_cmd(cmd, cwd=repo_root)


def main():
    parser = argparse.ArgumentParser(description="Auto-apply natural language SQL rules to Go code.")
    parser.add_argument("--rules", default=DEFAULT_RULES_FILE, help="Rules file (natural language).")
    parser.add_argument("--state", default=DEFAULT_STATE_FILE, help="State file for processed rules.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL), help="OpenAI model name.")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""), help="OpenAI API base URL.")
    parser.add_argument("--files", action="append", default=[], help="Extra Go files to include as context.")
    parser.add_argument("--dry-run", action="store_true", help="Only print diff, do not apply.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip build and scenario run.")
    parser.add_argument("--scenarios-args", default="", help="Extra args for cmd/scenarios/run-scenarios.")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(repo_root)

    rules_path = args.rules
    if not os.path.isabs(rules_path):
        rules_path = os.path.join(repo_root, rules_path)
    if not os.path.exists(rules_path):
        print(f"Rules file not found: {rules_path}", file=sys.stderr)
        sys.exit(1)

    state_path = args.state
    if not os.path.isabs(state_path):
        state_path = os.path.join(repo_root, state_path)

    rules_text = read_text(rules_path)
    rules = split_rules(rules_text)
    state = load_state(state_path)
    new_rules = get_new_rules(rules, state)
    if not new_rules:
        print("No new rules found.")
        return

    code_files = collect_files(DEFAULT_CODE_FILES + args.files)
    if not code_files:
        print("No code files found for context.", file=sys.stderr)
        sys.exit(1)

    system_prompt = (
        "You are a senior Go engineer. Your task is to implement the new natural language SQL rules into "
        "the existing Go codebase.\n\n"
        "Requirements:\n"
        "1) Output unified diff only (no explanations, no markdown).\n"
        "2) Modify only the provided files unless absolutely necessary.\n"
        "3) Do not remove or change existing behavior unless required by the new rules.\n"
        "4) Ensure code compiles; add missing imports if needed.\n"
        "5) If you cannot safely implement, return an empty diff.\n"
    )

    rules_block = "\n\n".join([f"[Rule {i+1}]\n{r}" for i, (_, r) in enumerate(new_rules)])
    user_prompt = (
        "New natural language rules:\n"
        f"{rules_block}\n\n"
        "Project files:\n"
        f"{format_files_for_prompt(code_files)}\n"
        "Return unified diff only."
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = args.api_base or None
    diff = openai_chat(api_key, args.model, system_prompt, user_prompt, api_base=api_base)

    if args.dry_run:
        print(diff)
        return

    apply_diff(diff)
    changed_go = extract_changed_go_files(diff)
    gofmt_files([os.path.join(repo_root, p) for p in changed_go])

    if not args.skip_tests:
        scenario_args = shlex.split(args.scenarios_args) if args.scenarios_args else []
        build_and_test(repo_root, scenario_args)

    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for h, r in new_rules:
        state.setdefault("rules", {})[h] = {
            "rule": r,
            "applied_at": now,
            "status": "applied",
        }
    save_state(state_path, state)
    print(f"Applied {len(new_rules)} new rule(s).")


if __name__ == "__main__":
    main()
