## GPU/CUDA access

- This repo uses the `/data/Andy/fcclip-torch27-cu126.sif` Apptainer image for GPU/PyTorch work.
- In the default Codex sandbox, NVIDIA devices may be hidden: `nvidia-smi` can fail and `torch.cuda.is_available()` can return `False` even though the host GPU works.
- For GPU checks or CUDA workloads, run commands outside the sandbox with escalation. The useful approval prefixes are:
  - `nvidia-smi`
  - `apptainer exec --nv`
- Build the CUDA extension with:

```bash
apptainer exec --nv \
  --bind "$PWD:/workspace/synthetic-scene" \
  --pwd /workspace/synthetic-scene \
  /data/Andy/fcclip-torch27-cu126.sif \
  python setup.py build_ext --inplace
```
