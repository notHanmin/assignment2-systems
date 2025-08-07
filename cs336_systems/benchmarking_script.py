import sys
import os
import torch
import timeit
import numpy as np
from adapters import TransformerLM, AdamW

def run_benchmark(model, optimizer, device, warmup_step, running_step, batch_size, context_length):

    model.to(device)
    model.train()

    input_data = torch.randint(0, model.vocab_size, (batch_size, context_length), device=device)

    timings = []
    
    for step in range(warmup_step + running_step):
        torch.cuda.synchronize()
        start_time = timeit.default_timer()

        logits = model(input_data)

        loss = logits.sum()
        optimizer.zero_grad()
        loss.backward()

        torch.cuda.synchronize()

        end_time = timeit.default_timer()

        # Only record the timings for the measurement steps
        if step >= warmup_step:
            timings.append(end_time - start_time)

    return np.mean(timings), np.std(timings)

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 10000
    batch_size = 4
    warmup_steps = [0,1,2,5]
    running_step = 10

    model_params = {
        # d_model d_ff num_layers num_heads context_len
        'small_128' : [768, 3072, 12, 12, 128],
        'small_256' : [768, 3072, 12, 12, 256],
        'medium_128' : [1024, 4096, 24, 16, 128],
        'medium_256' : [1024, 4096, 24, 16, 256],
        'large_128' : [1280, 5120, 36, 20, 128],
        'large_256' : [1280, 5120, 36, 20, 256],
        'xl_128' : [1600, 6400, 48, 25, 128],
        'xl_256' : [1600, 6400, 48, 25, 256],
        '2.7b_128' : [2560, 10240, 32, 32, 128],
        '2.7b_256' : [2560, 10240, 32, 32, 256]
    }

    for size, params in model_params.items():

        model = TransformerLM(
        vocab_size=vocab_size,
        context_length=params[4],
        d_model=params[0],
        num_layers=params[2],
        num_heads=params[3],
        d_ff=params[1],
        rope_theta=10000,
        device=device
        )

        model.to(device)
        optimizer = AdamW(model.parameters())

        print(f"Model parameter: {size}")

        for warmup_step in warmup_steps:
            avg, std = run_benchmark(model, optimizer, device, warmup_step, running_step, batch_size, params[4])
            print(f"Warm up step count: {warmup_step} | Avg time: {avg} | STD: {std}" )

if __name__ == "__main__":
    main()