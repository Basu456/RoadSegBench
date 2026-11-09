import time, numpy as np, torch
def measure_latency(model, input_shape=(1,3,512,1024), warmup=50, iterations=200,
                    precision="fp32", backend="pytorch"):
    if backend!="pytorch":
        raise NotImplementedError("This function measures PyTorch only. Implement and report each backend separately.")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device).eval()
    x=torch.randn(*input_shape,device=device)
    use_amp=precision in {"fp16","bf16"} and device=="cuda"
    dtype=torch.float16 if precision=="fp16" else torch.bfloat16
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast("cuda",dtype=dtype,enabled=use_amp): model(x)
        if device=="cuda": torch.cuda.synchronize()
        vals=[]
        for _ in range(iterations):
            t=time.perf_counter()
            with torch.autocast("cuda",dtype=dtype,enabled=use_amp): model(x)
            if device=="cuda": torch.cuda.synchronize()
            vals.append((time.perf_counter()-t)*1000)
    return {"device":device,"backend":backend,"precision":precision,
            "warmup":warmup,"iterations":iterations,
            "mean_ms":float(np.mean(vals)),"median_ms":float(np.median(vals)),
            "p95_ms":float(np.percentile(vals,95)),"fps_batch1":float(1000/np.mean(vals))}
