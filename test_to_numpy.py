#!/usr/bin/env python3
"""Test to_numpy_4d signature"""
import sys
import importlib

# Complete clean slate
for mod in list(sys.modules.keys()):
    if 'functions' in mod or 'read_from_file' in mod:
        del sys.modules[mod]

importlib.invalidate_caches()

# Fresh import
from functions import to_numpy_4d
import inspect

sig = inspect.signature(to_numpy_4d)
print("SUCCESS! to_numpy_4d signature is:")
print(f"  {sig}")
print(f"\nParameters: {list(sig.parameters.keys())}")
print(f"Defaults: {[(k, v.default) for k, v in sig.parameters.items() if v.default != inspect.Parameter.empty]}")

# Check gen parameter specifically
if 'gen' in sig.parameters:
    gen_param = sig.parameters['gen']
    print(f"\n✓ gen parameter found! Default value: {gen_param.default}")
else:
    print("\n✗ ERROR: gen parameter NOT found!")
    sys.exit(1)
