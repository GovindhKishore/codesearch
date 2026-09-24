from dataclasses import dataclass, field
import ast, os
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
    callees: list[tuple[str, str | None]]
    callers: list[tuple[str, str | None]] = field(default_factory=list)
    class_name: str | None = None
    composite_doc: str = field(default="", repr=False)

class CodebaseParser:

    def __init__(self, skip_dirs: set[str] | None = None):
        self.skip_dirs = skip_dirs or {
            "__pycache__", ".git", "venv", ".venv", "node_modules", "dist", "build",
            ".eggs", "egg-info", ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
            ".tox", "htmlcov",
        }

    def parse_dir(self, folder: Path) -> list[FunctionInfo]:
        all_functions = []
        name_fninfo_map = defaultdict(list)

        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames[:] = [d for d in dirnames if d not in self.skip_dirs]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                filepath = Path(dirpath) / filename
                file_functions = self.parse_file(filepath)
                all_functions.extend(file_functions)
                for func in file_functions:
                    name_fninfo_map[(func.name, func.class_name)].append(func)


        for func in all_functions:
            for callee_name, callee_class in func.callees:
                for callee_func in name_fninfo_map[(callee_name, callee_class)]:
                    if callee_func is not func:
                        callee_func.callers.append((func.name, func.class_name))


        for func in all_functions:
            func.composite_doc = self.build_composite_doc(func)

        return all_functions

    def parse_file(self, filepath: Path) -> list[FunctionInfo]:
        try:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, ValueError, UnicodeDecodeError, OSError):
            return []

        functions: list[FunctionInfo] = []
        self._walk_body(tree.body, filepath, None, functions)
        return functions

    @staticmethod
    def build_composite_doc(func: FunctionInfo) -> str:
        parts = [func.name.replace("_", " ")]

        if func.doc_string:
            parts.append(func.doc_string)

        if func.params:
            normalized_params = [p.replace("_", " ") for p in func.params]
            parts.append(" ".join(normalized_params))

        if func.return_type:
            parts.append(func.return_type.replace("_", " "))

        if func.callees:
            normalized_callees = [c + callee.replace("_", " ") if c is not None else callee.replace("_", " ") for callee, c in func.callees]
            parts.append(" ".join(normalized_callees))

        if func.callers:
            normalized_callers = [c + caller.replace("_", " ") if c is not None else caller.replace("_", " ") for caller, c in func.callers]
            parts.append(" ".join(normalized_callers))

        module = str(func.file).replace("\\", "/")
        module = module.replace(".py", "").replace("/", " ").replace("_", " ")
        parts.append(module)

        return "\n".join(parts)

    @staticmethod
    def _extract_params(args_node: ast.arguments) -> list[str]:
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

    @staticmethod
    def _extract_callees(func_node: ast.FunctionDef | ast.AsyncFunctionDef, current_class: str | None) -> list[tuple[str, str | None]]:
        callees = set()

        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue

            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name not in BUILTIN_SKIP:
                    callees.add((name, None))

            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if name in BUILTIN_SKIP:
                    continue
                if isinstance(node.func.value, ast.Name) and node.func.value.id in ("self", "cls"):
                    callees.add((name, current_class))
                else:
                    callees.add((name, "UNKNOWN"))

        return list(callees)

    def _walk_body(self, body: list[ast.stmt], filepath : Path, current_class: str | None, functions: list[FunctionInfo]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                file = filepath
                line = node.lineno
                doc_string = ast.get_docstring(node)
                params = self._extract_params(node.args)
                return_type = ast.unparse(node.returns) if node.returns else None
                callees = self._extract_callees(node, current_class)

                functions.append(FunctionInfo(
                    name=name,
                    file=file,
                    line=line,
                    doc_string=doc_string,
                    params=params,
                    return_type=return_type,
                    callees=callees,
                    class_name=current_class,
                ))
                self._walk_body(node.body, filepath, current_class, functions)
            elif isinstance(node, ast.ClassDef):
                self._walk_body(node.body, filepath, node.name, functions)
            elif isinstance(node, ast.Try):
                self._walk_body(node.body, filepath, current_class, functions)
                for handler in node.handlers:
                    self._walk_body(handler.body, filepath, current_class, functions)
                self._walk_body(node.orelse, filepath, current_class, functions)
                self._walk_body(node.finalbody, filepath, current_class, functions)
            elif hasattr(node, "body"):
                self._walk_body(node.body, filepath, current_class, functions)
                if hasattr(node, "orelse"):
                    self._walk_body(node.orelse, filepath, current_class, functions)




