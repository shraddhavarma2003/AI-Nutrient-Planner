import os
import ast

def extract_purpose(file_path: str) -> str:
    """Return a short description of the given Python file.
    Preference order:
    1. Module-level docstring.
    2. First top-level comment block (lines starting with '#').
    3. Heuristic based on filename.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception:
        return "Unable to read file"

    # Try to parse AST for docstring
    try:
        module = ast.parse(source)
        doc = ast.get_docstring(module)
        if doc:
            # Return first line of docstring as summary
            return doc.strip().splitlines()[0]
    except Exception:
        pass

    # Fallback: look for top-level comments at start of file
    lines = source.splitlines()
    comment_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            comment_lines.append(stripped.lstrip("# "))
        elif stripped == "":
            continue
        else:
            break
    if comment_lines:
        return " ".join(comment_lines).strip()

    # Heuristic based on filename
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    # Simple heuristics: replace underscores with spaces and add "module"
    words = name.replace("_", " ")
    return f"{words.capitalize()} module"
