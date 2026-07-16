from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def load_prompt(name: str) -> str:
    prompt_path = PROMPTS_DIR / f"{name}.prompt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def load_prompt_or_default(name: str, default: str) -> str:
    content = load_prompt(name)
    return content if content else default
