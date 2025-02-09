import time
import numpy as np
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput

def response_callback(user_data, result, error):
    if error:
        user_data["error"] = error
    else:
        # Record the time when the first chunk is received
        if user_data["first_chunk_time"] is None:
            user_data["first_chunk_time"] = time.perf_counter()
        # Assuming the output is a bytes array representing the audio chunk
        audio_chunk = result.as_numpy("OUT")[0]
        # Append the audio chunk to the list
        user_data["audio_chunks"].append(audio_chunk)

def perform_inference(url, model_name, input_text):
    # Create a new gRPC client for each inference
    try:
        triton_client = grpcclient.InferenceServerClient(url=url)
    except Exception as e:
        print(f"Failed to create Triton client: {e}")
        return None

    # Prepare the input tensor
    inputs = [
        InferInput("TEXT", [1], "BYTES")
    ]
    inputs[0].set_data_from_numpy(np.array([input_text], dtype=object))

    # Prepare the output tensor
    outputs = [
        InferRequestedOutput("OUT")
    ]

    # Dictionary to store audio chunks, timing information, and potential errors
    user_data = {
        "audio_chunks": [],
        "error": None,
        "first_chunk_time": None
    }

    # Record the start time of the request
    request_start_time = time.perf_counter()

    # Start the streaming inference
    try:
        # Start the stream with the callback function
        triton_client.start_stream(callback=partial(response_callback, user_data=user_data))

        # Send the inference request
        triton_client.async_stream_infer(model_name=model_name, inputs=inputs, outputs=outputs)

        # Signal the end of requests
        triton_client.stop_stream()

        # Check for errors
        if user_data["error"]:
            print(f"Inference failed: {user_data['error']}")
            return None

    except grpcclient.InferenceServerException as e:
        print(f"Inference failed: {e}")
        return None

    # Record the end time of the response
    response_end_time = time.perf_counter()

    # Calculate Time to First Chunk (TTFC)
    if user_data["first_chunk_time"]:
        ttfc = user_data["first_chunk_time"] - request_start_time
    else:
        print("No audio chunks received.")
        return None

    # Combine all audio chunks
    complete_audio = b''.join(user_data["audio_chunks"])

    # Calculate the duration of the generated audio in seconds
    # Assuming 16-bit (2 bytes) samples and a sample rate of 22050 Hz
    audio_duration = len(complete_audio) / (2 * 22050)

    # Calculate the total processing time
    total_processing_time = response_end_time - request_start_time

    # Calculate Real-Time Factor (RTF)
    rtf = total_processing_time / audio_duration

    # Return the metrics and audio data
    return {
        "ttfc": ttfc,
        "rtf": rtf,
        "audio_data": complete_audio
    }

def benchmark_concurrency(url, model_name, input_text, max_concurrency=10):
    concurrency_levels = range(1, max_concurrency + 1)
    ttfc_results = []
    rtf_results = []

    for concurrency in concurrency_levels:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(perform_inference, url, model_name, input_text) for _ in range(concurrency)]
            ttfc_sum = 0
            rtf_sum = 0
            completed_requests = 0

            for future in as_completed(futures):
                result = future.result()
                if result:
                    ttfc_sum += result['ttfc']
                    rtf_sum += result['rtf']
                    completed_requests += 1

            if completed_requests > 0:
                avg_ttfc = ttfc_sum / completed_requests
                avg_rtf = rtf_sum / completed_requests
            else:
                avg_ttfc = float('nan')
                avg_rtf = float('nan')

            ttfc_results.append(avg_ttfc)
            rtf_results.append(avg_rtf)

            print(f"Concurrency Level: {concurrency}, Average TTFC: {avg_ttfc:.4f} seconds, Average RTF: {avg_rtf:.4f}")

    # Plotting the results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(concurrency_levels, ttfc_results, marker='o')
    plt.title('Average TTFC vs. Concurrency Level')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Average TTFC (seconds)')

    plt.subplot(1, 2, 2)
    plt.plot(concurrency_levels, rtf_results, marker='o', color='orange')
    plt.title('Average RTF vs. Concurrency Level')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Average RTF')

    plt.tight_layout()
    plt.savefig('benchmark_results.png')  # Save the plot as a PNG file
    plt.show()

def main():
    # Server URL
    url = "localhost:8001"  # Adjust if your server is running elsewhere

    # Model name
    model_name = "xtts_v2"  # Ensure this matches your deployed model's name

    # Input text for inference
    input_text = "Hello, I am Akash Sonowal"

    # Perform benchmarking
    benchmark_concurrency(url, model_name, input_text)

if __name__ == "__main__":
    main()