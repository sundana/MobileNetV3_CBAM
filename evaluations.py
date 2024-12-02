import torch
import time

def measure_throughput_latency(model, dataloader, device='cuda'):
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # For timing
    batch_latency_times = []
    total_samples = 0
    total_time = 0.0

    # Warm-up (especially important for GPU timing)
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            model(inputs)
            break

    # Measure throughput and latency
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            total_samples += inputs.size(0)

            # Start timing
            start_time = time.time()
            model(inputs)  # Forward pass
            end_time = time.time()

            # Record latency for this batch
            batch_latency = end_time - start_time
            batch_latency_times.append(batch_latency)

            # Accumulate total time
            total_time += batch_latency

    # Calculate metrics
    avg_latency = sum(batch_latency_times) / len(batch_latency_times)
    throughput = total_samples / total_time  # in samples per second

    return avg_latency, throughput