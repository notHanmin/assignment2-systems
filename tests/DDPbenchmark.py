import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import numpy as np
import einops

current_dir = os.path.dirname(__file__)
training_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'assignment1-basics', 'tests'))
data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'assignment1-basics', 'tests', 'fixtures'))
sys.path.insert(0, training_dir)
sys.path.insert(1, data_dir)
import training
from training import Tokenizer, AdamW, TransformerLM, run_cross_entropy, run_get_batch


def _print_allreduce_benchmark(rank, label, worldSize, measured_steps, allreduce_calls, allreduce_time_ms):
    stats = torch.tensor(
        [float(measured_steps), float(allreduce_calls), float(allreduce_time_ms)],
        dtype=torch.float64,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    if rank == 0:
        max_measured_steps = int(stats[0].item())
        max_calls = int(stats[1].item())
        max_total_ms = stats[2].item()
        avg_ms_per_call = (max_total_ms / max_calls) if max_calls > 0 else 0.0
        avg_ms_per_step = (max_total_ms / max_measured_steps) if max_measured_steps > 0 else 0.0
        print(
            f"{label} | world_size={worldSize} | warmup_steps=5 | "
            f"measured_steps={max_measured_steps} | all_reduce_calls={max_calls} | "
            f"total_all_reduce_ms={max_total_ms:.4f} | avg_ms/call={avg_ms_per_call:.4f} | "
            f"avg_ms/step={avg_ms_per_step:.4f}"
        )

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, DataSizes):
    setup(rank, world_size)
    
    for DataSize in DataSizes:
        data = torch.randint(0, 10, (DataSize,))

        for _ in range(5):
            dist.all_reduce(data, async_op=False)

        dist.barrier() 
        
        start_time = time.perf_counter()
        dist.all_reduce(data, async_op=False)
        end_time = time.perf_counter()
        
        elapsed_time_ms = (end_time - start_time) * 1000
        
        if rank == 0:
            print(f"Gloo | World Size: {world_size} | Data Size: {DataSize} | Time: {elapsed_time_ms:.4f} ms")

def setup_nccl(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def distributed_demo_nccl(rank, world_size, DataSizes):
    torch.cuda.set_device(rank)
    setup_nccl(rank, world_size)
    
    for DataSize in DataSizes:
        data = torch.randint(0, 10, (DataSize,)).to(rank)

        for _ in range(5):
            dist.all_reduce(data, async_op=False)

        dist.barrier() 
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        dist.all_reduce(data, async_op=False) # <--- The actual work
        end_event.record()
        
        torch.cuda.synchronize() # Wait for GPU to finish
        
        local_time_ms = start_event.elapsed_time(end_event)
        
        time_tensor = torch.tensor([local_time_ms], device=rank)
        
        dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
        
        if rank == 0:
            max_time = time_tensor.item()
            print(f"NCCL | World Size: {world_size} | Data Size: {DataSize} | Time: {max_time:.4f} ms")

    dist.destroy_process_group()

def training_nccl(rank, worldSize, args):
    torch.cuda.set_device(rank)
    setup_nccl(rank, worldSize)

    model = training.TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=rank  # <--- Important: Build on the specific GPU
    )

def training_gloo(rank, worldSize, args):
    setup(rank, worldSize)
    try:
        device = "cpu"
        warmup_steps = 5
        measured_steps = 0
        allreduce_calls = 0
        allreduce_time_ms = 0.0
        # tiny data to avoid memory
        sample = "hello world. " * 200
        tokenizer = training.Tokenizer.from_files(
            os.path.join(data_dir, "gpt2_vocab.json"),
            os.path.join(data_dir, "gpt2_merges.txt"),
            special_tokens=["<|endoftext|>"],
        )
        token_ids = np.fromiter(tokenizer.encode_iterable(sample.splitlines()), dtype=np.uint16)
        train_data = token_ids  # skip np.save/load roundtrip
        # model small
        model = training.TransformerLM(
            vocab_size=len(tokenizer.vocab),
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            device=device,
        )
        model.to(device)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        for step in range(args.num_training_steps):
            if step == warmup_steps:
                dist.barrier()
            x, y = run_get_batch(train_data, args.batch_size, args.context_length, device=device)
            logits = model(x)
            logits_r = einops.rearrange(logits, "b s v -> (b s) v")
            targets_r = einops.rearrange(y, "b s -> (b s)")
            loss = run_cross_entropy(logits_r, targets_r)
            optimizer.zero_grad()
            loss.backward()
            timed_step = step >= warmup_steps
            if timed_step:
                measured_steps += 1
            for p in model.parameters():
                if timed_step:
                    start_time = time.perf_counter()
                    dist.all_reduce(p.grad)
                    end_time = time.perf_counter()
                    allreduce_time_ms += (end_time - start_time) * 1000.0
                    allreduce_calls += 1
                else:
                    dist.all_reduce(p.grad)
                p.grad.div_(worldSize)
            optimizer.step()
            if rank == 0 and step % args.log_interval == 0:
                print("Step", step, "loss", loss.item())
        _print_allreduce_benchmark(
            rank,
            label="training_gloo",
            worldSize=worldSize,
            measured_steps=measured_steps,
            allreduce_calls=allreduce_calls,
            allreduce_time_ms=allreduce_time_ms,
        )
    finally:
        dist.destroy_process_group()

def training_gloo_flat(rank, worldSize, args):
    setup(rank, worldSize)
    try:
        device = "cpu"
        warmup_steps = 5
        measured_steps = 0
        allreduce_calls = 0
        allreduce_time_ms = 0.0
        # tiny data to avoid memory
        sample = "hello world. " * 200
        tokenizer = training.Tokenizer.from_files(
            os.path.join(data_dir, "gpt2_vocab.json"),
            os.path.join(data_dir, "gpt2_merges.txt"),
            special_tokens=["<|endoftext|>"],
        )
        token_ids = np.fromiter(tokenizer.encode_iterable(sample.splitlines()), dtype=np.uint16)
        train_data = token_ids  # skip np.save/load roundtrip
        # model small
        model = training.TransformerLM(
            vocab_size=len(tokenizer.vocab),
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            device=device,
        )
        model.to(device)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        for step in range(args.num_training_steps):
            if step == warmup_steps:
                dist.barrier()
            x, y = run_get_batch(train_data, args.batch_size, args.context_length, device=device)
            logits = model(x)
            logits_r = einops.rearrange(logits, "b s v -> (b s) v")
            targets_r = einops.rearrange(y, "b s -> (b s)")
            loss = run_cross_entropy(logits_r, targets_r)
            optimizer.zero_grad()
            loss.backward()
            grads = [p.grad for p in model.parameters()]
            flat = torch._utils._flatten_dense_tensors(grads)
            timed_step = step >= warmup_steps
            if timed_step:
                measured_steps += 1
                start_time = time.perf_counter()
                dist.all_reduce(flat)
                end_time = time.perf_counter()
                allreduce_time_ms += (end_time - start_time) * 1000.0
                allreduce_calls += 1
            else:
                dist.all_reduce(flat)
            flat.div_(worldSize)
            unflattened = torch._utils._unflatten_dense_tensors(flat, grads)
            for param, new_grad in zip(model.parameters(), unflattened):
                param.grad.copy_(new_grad)
            optimizer.step()
            if rank == 0 and step % args.log_interval == 0:
                print("Step", step, "loss", loss.item())
        _print_allreduce_benchmark(
            rank,
            label="training_gloo_flat",
            worldSize=worldSize,
            measured_steps=measured_steps,
            allreduce_calls=allreduce_calls,
            allreduce_time_ms=allreduce_time_ms,
        )
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    parser.add_argument('--world_sizes', nargs='+', type=int, default=[1])
    parser.add_argument('--num_training_steps', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--context_length', type=int, default=64)
    parser.add_argument('--rope_theta', type=int, default=10000, help='RoPE theta.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='The learning rate for the optimizer.')
    parser.add_argument('--log_interval', type=int, default=2, help='How often to log training loss.')
    parser.add_argument(
        '--benchmark_mode',
        type=str,
        default='both',
        choices=['gloo', 'flat', 'both'],
        help='Which benchmark to run.',
    )

    args = parser.parse_args()

    for w in args.world_sizes:
        if args.benchmark_mode in ('gloo', 'both'):
            mp.spawn(fn=training_gloo, args=(w, args), nprocs=w, join=True)
        if args.benchmark_mode in ('flat', 'both'):
            mp.spawn(fn=training_gloo_flat, args=(w, args), nprocs=w, join=True)
        