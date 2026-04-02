
import config.static_vars
print("Imported config.static_vars")
print(dir(config.static_vars))
try:
    print(f"block_path: {config.static_vars.block_path}")
except AttributeError:
    print("block_path not found in config.static_vars")
