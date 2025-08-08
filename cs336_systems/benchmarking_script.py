import sys
import os
import torch
import timeit
import numpy as np
import argparse
from adapters import TransformerLM, AdamW
import torch.cuda.nvtx as nvtx

def run_benchmark(model, optimizer, device, warmup_steps, running_steps, batch_size, context_length):
    """
    Runs benchmarking for a given model configuration, timing forward and backward
    passes separately.

    Args:
        model (torch.nn.Module): The Transformer model to benchmark.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (str): The device to run on ('cuda' or 'cpu').
        warmup_steps (int): The number of warm-up steps to run.
        running_steps (int): The number of steps to measure and average.
        batch_size (int): The batch size for the input data.
        context_length (int): The context length for the input data.

    Returns:
        tuple: A tuple containing (avg_fwd, std_fwd, avg_bwd, std_bwd).
    """
    model.to(device)
    model.train()

    autocast_context = torch.autocast(device_type=device, dtype=dtype)

    # Generate a single, reusable batch of random data
    input_data = torch.randint(0, model.vocab_size, (batch_size, context_length), device=device)

    forward_timings = []
    backward_timings = []
    
    for step in range(warmup_steps + running_steps):
        # --- Time Forward Pass ---
        torch.cuda.synchronize()
        start_fwd = timeit.default_timer()
        
        with autocast_context:
            logits = model(input_data)
        
        torch.cuda.synchronize()
        end_fwd = timeit.default_timer()
        
        if step >= warmup_steps:
            forward_timings.append(end_fwd - start_fwd)

        # --- Time Backward Pass ---
        # Prepare for backward pass
        loss = logits.sum()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        start_bwd = timeit.default_timer()

        loss.backward()

        torch.cuda.synchronize()
        end_bwd = timeit.default_timer()

        if step >= warmup_steps:
            backward_timings.append(end_bwd - start_bwd)

    # Calculate statistics for each pass
    avg_fwd = np.mean(forward_timings) if forward_timings else 0
    std_fwd = np.std(forward_timings) if forward_timings else 0
    avg_bwd = np.mean(backward_timings) if backward_timings else 0
    std_bwd = np.std(backward_timings) if backward_timings else 0

    return avg_fwd, std_fwd, avg_bwd, std_bwd

def run_benchmark_with_nvtx(model, optimizer, device, warmup_steps, running_steps, batch_size, context_length):
    
    model.to(device)
    model.train()

    dtype = torch.bfloat16
    autocast_context = torch.autocast(device_type=device, dtype=dtype)

    input_data = torch.randint(0, model.vocab_size, (batch_size, context_length), device=device)
    timings = []

    # --- 1. Warm-up Loop (Not in an NVTX range) ---
    # This part runs as normal, but its operations won't be in our "Measurement" range.
    print(f"Running {warmup_steps} warm-up steps...")
    for _ in range(warmup_steps):
        logits = model(input_data)
        loss = logits.sum()
        optimizer.zero_grad()
        loss.backward()
    
    # Ensure all warm-up work is done before starting the measurement range
    torch.cuda.synchronize()

    # --- 2. Measurement Loop (Wrapped in an NVTX range) ---
    print(f"Running {running_steps} measurement steps within NVTX range...")
    with nvtx.range("Measurement Steps"):
        for _ in range(running_steps):
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            torch.cuda.memory._record_memory_history(max_entries=1000000)
            with nvtx.range("Forward pass"):
                logits = model(input_data)
                torch.cuda.memory._dump_snapshot("memory_snapshot_forward_without_mixed.pickle")
                #with autocast_context:
                    #logits = model(input_data)
                    #torch.cuda.memory._dump_snapshot("memory_snapshot_forward.pickle")
                loss = logits.sum()
                

            with nvtx.range("Backward pass"):
                optimizer.zero_grad()
                loss.backward()

            with nvtx.range("Optimizer step"):
                optimizer.step()
            torch.cuda.memory._dump_snapshot("memory_snapshot_full_without_mixed.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            
            timings.append(end_time - start_time)

    return np.mean(timings), np.std(timings)

<<<<<<< HEAD
def run_benchmark_attn(model, optimizer, device, warmup_steps, running_steps, batch_size, context_length):
    model.to(device)
    model.train()
    input_data = torch.randint(0, model.vocab_size, (batch_size, context_length), device=device)
    timings = []

    print(f"Running {warmup_steps} warm-up steps...")
    for _ in range(warmup_steps):
        logits = model(input_data)
        loss = logits.sum()
        optimizer.zero_grad()
        loss.backward()

    print(f"Running {running_steps} measurement steps within NVTX range...")
    with nvtx.range("Measurement Steps"):
        for _ in range(running_steps):
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            torch.cuda.memory._record_memory_history(max_entries=1000000)
            with nvtx.range("Forward pass"):
                logits = model(input_data)
                loss = logits.sum()
                

            with nvtx.range("Backward pass"):
                optimizer.zero_grad()
                loss.backward()

            with nvtx.range("Optimizer step"):
                optimizer.step()
            torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            
            timings.append(end_time - start_time)

    return np.mean(timings), np.std(timings)
=======
def benchmark_attn(model, optimizer, device):

>>>>>>> 3331f50 (Generate snapshots)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 1
    batch_size = 4
    warmup_configs = [0, 1, 2, 5]
    running_steps = 10

    model_params = {
        # d_model, d_ff, num_layers, num_heads, context_length
        'xl_256' : [1600, 6400, 48, 25, 256]

        # 'small_128' : [768, 3072, 12, 12, 128],
        # 'small_256' : [768, 3072, 12, 12, 256],
        # 'medium_128' : [1024, 4096, 24, 16, 128],
        # 'medium_256' : [1024, 4096, 24, 16, 256],
        # 'large_128' : [1280, 5120, 36, 20, 128],
        # 'large_256' : [1280, 5120, 36, 20, 256],
        # 'xl_128' : [1600, 6400, 48, 25, 128],
        # 'xl_256' : [1600, 6400, 48, 25, 256]
        #'2.7b_128' : [2560, 10240, 32, 32, 128],
        #'2.7b_256' : [2560, 10240, 32, 32, 256]
    }

    print("--- Profiling Forward and Backward Passes ---")

    for size, params in model_params.items():
        d_model, d_ff, num_layers, num_heads, context_length = params
        model = TransformerLM(
            vocab_size=vocab_size, context_length=context_length, d_model=d_model,
            num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=10000
        )
        optimizer = AdamW(model.parameters())

        print(f"\nModel Configuration: {size}")
        print("-" * 60)

        for num_warmup in warmup_configs:
            #avg_fwd, std_fwd, avg_bwd, std_bwd = run_benchmark_with_nvtx(
            avg, std = run_benchmark_attn(
                model, optimizer, device,
                num_warmup, running_steps, batch_size, context_length
            )
            #print(
                #f"Warm-up Steps: {num_warmup:<2} | "
                #f"Forward Avg: {avg_fwd:.6f}s (Std: {std_fwd:.6f}) | "
                #f"Backward Avg: {avg_bwd:.6f}s (Std: {std_bwd:.6f})"
            #)

if __name__ == "__main__":
    main()
