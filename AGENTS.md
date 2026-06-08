## GPU/CUDA access

- This repo uses the `clipdino-cu117` conda environment for GPU/PyTorch work.
- In the default Codex sandbox, NVIDIA devices may be hidden: `nvidia-smi` can fail and `torch.cuda.is_available()` can return `False` even though the host GPU works.
- For GPU checks or CUDA workloads, run commands outside the sandbox with escalation. The useful approval prefixes are:
  - `nvidia-smi`
  - `conda run -n clipdino-cu117`
- Verified outside the sandbox: `conda run -n clipdino-cu117` can access CUDA in PyTorch, with one visible `NVIDIA GeForce RTX 2080 SUPER` device.
