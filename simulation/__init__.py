"""Simulation package; EGL is the portable headless default for CI/WSL."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
