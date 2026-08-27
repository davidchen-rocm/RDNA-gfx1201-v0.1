import ast
import re
from pathlib import Path

root = Path("/home/david/Desktop/aiter-q4rdna")
cpp = (root / "csrc/kernels/q4_group64_gemv.cu").read_text()
python = (root / "aiter/ops/q4_group64_gemv.py").read_text()

cache_body = re.search(
    r"std::string cached_gpu_arch\(int device_id\)\n\{(?P<body>.*?)\n\}",
    cpp,
    flags=re.DOTALL,
)
assert cache_body is not None
body = cache_body.group("body")
assert "std::unordered_map<int, std::string> cache" in body
assert "cache.find(device_id)" in body
assert "cache.emplace(device_id, arch)" in body
assert "get_gpu_arch()" in body

entry = cpp.index("void q4_group64_gemv_out")
guard = cpp.index("HipDeviceGuard device_guard(x.device_id);", entry)
lookup = cpp.index("cached_gpu_arch(x.device_id)", entry)
assert guard < lookup, "device selection/validation must precede the keyed cache lookup"

module = ast.parse(python)
arch_function = next(
    node
    for node in module.body
    if isinstance(node, ast.FunctionDef) and node.name == "_gfx_arch_for_index"
)
assert any(
    isinstance(decorator, ast.Name) and decorator.id == "cache"
    for decorator in arch_function.decorator_list
)
properties_call = next(
    node
    for node in ast.walk(arch_function)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "get_device_properties"
)
assert isinstance(properties_call.args[0], ast.Name)
assert properties_call.args[0].id == "index"

print("per-device cache audit passed: valid device selection precedes keyed lookup")
