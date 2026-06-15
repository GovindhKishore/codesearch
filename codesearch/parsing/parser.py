from dataclasses import dataclass, field
import ast
from pathlib import Path
from collections import defaultdict

BUILTIN_SKIP = {
    "print", "len", "range", "str", "int", "float", "bool",
    "list", "dict", "set", "tuple", "isinstance", "hasattr",
    "getattr", "setattr", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "any", "all", "sum", "min", "max",
    "open", "super", "type", "repr", "format", "append"
}

@dataclass
class FunctionInfo:
    name: str
    file: Path
    line: int
    doc_string: str | None
    params: list[str]
    return_type: str | None
    callees: list[str]
    callers: list[str] = field(default_factory=list)
    composite_doc: str = field(default="", repr=False)

class CodebaseParser:

    def __init__(self, skip_dirs: list[str] | None = None):
        self.skip_dirs = skip_dirs or [
            "__pycache__",
            ".git",
            "venv",
            ".venv",
            "node_modules",
            "dist",
            "build",
            ".eggs",
            "egg-info",
        ]

    def parse_dir(self, folder: Path) -> list[FunctionInfo]:
        all_functions = []
        name_fninfo_map = defaultdict(list)

        for filepath in folder.rglob("*.py"):
            if any(skip in filepath.parts for skip in self.skip_dirs):
                continue
            file_functions = self.parse_file(filepath)
            all_functions.extend(file_functions)
            for func in file_functions:
                name_fninfo_map[func.name].append(func)


        for func in all_functions:
            for callee_name in func.callees:
                for callee_func in name_fninfo_map[callee_name]:
                    if callee_func is not func:
                        callee_func.callers.append(func.name)


        for func in all_functions:
            func.composite_doc = self.build_composite_doc(func)

        return all_functions

    def parse_file(self, filepath: Path) -> list[FunctionInfo]:
        try:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, ValueError, UnicodeDecodeError, OSError):
            return []

        functions = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            name = node.name
            file = filepath
            line = node.lineno
            doc_string = ast.get_docstring(node)
            params = self._extract_params(node.args)
            return_type = ast.unparse(node.returns) if node.returns else None
            callees = self._extract_callees(node)

            functions.append(FunctionInfo(
                name=name,
                file=file,
                line=line,
                doc_string=doc_string,
                params=params,
                return_type=return_type,
                callees=callees
            ))

        return functions

    def build_composite_doc(self, func: FunctionInfo) -> str:
        parts = []

        parts.append(func.name.replace("_", " "))

        if func.doc_string:
            parts.append(func.doc_string)

        if func.params:
            normalized_params = [p.replace("_", " ") for p in func.params]
            parts.append(" ".join(normalized_params))

        if func.return_type:
            parts.append(func.return_type.replace("_", " "))

        if func.callees:
            normalized_callees = [c.replace("_", " ") for c in func.callees]
            parts.append(" ".join(normalized_callees))

        if func.callers:
            normalized_callers = [c.replace("_", " ") for c in func.callers]
            parts.append(" ".join(normalized_callers))

        module = str(func.file).replace("\\", "/")
        module = module.replace(".py", "").replace("/", " ").replace("_", " ")
        parts.append(module)

        return "\n".join(parts)


    def _extract_params(self, args_node: ast.arguments) -> list[str]:
        params = []
        all_args = args_node.posonlyargs + args_node.args + args_node.kwonlyargs

        for arg in all_args:
            if arg.arg in ("self", "cls"):
                continue
            if arg.annotation:
                params.append(f"{arg.arg} {ast.unparse(arg.annotation)}")
            else:
                params.append(arg.arg)

        return params

    def _extract_callees(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        callees = set()

        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue

            if name not in BUILTIN_SKIP and name not in ("self", "cls"):
                callees.add(name)

        return list(callees)




