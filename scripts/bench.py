"""Calibrate step time at the three ladder widths before the scientific run."""
import time

import torch

from src.decision.models import PlainWorld


def bench(width: int, steps: int = 40):
    device = "cuda"
    model = PlainWorld(width=width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(512, 2, device=device)
    h = torch.rand(512, device=device)
    b = torch.randint(0, 4, (512,), device=device)
    for _ in range(5):
        out = model(x)
        loss = 10 * torch.nn.functional.mse_loss(out[:, 0], h) + (
            torch.nn.functional.cross_entropy(out[:, 1:], b)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        out = model(x)
        loss = 10 * torch.nn.functional.mse_loss(out[:, 0], h) + (
            torch.nn.functional.cross_entropy(out[:, 1:], b)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    print(
        f"w={width:4d} params={sum(p.numel() for p in model.parameters()):8d} "
        f"{dt*1000:7.2f} ms/step  20k={dt*20000/60:5.1f} min  "
        f"60k={dt*60000/60:5.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.get_device_name(0))
    for width in (256, 512, 1024, 1448):
        bench(width)
