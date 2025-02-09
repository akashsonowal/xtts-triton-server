import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput
import wave
import time
from functools import partial

def response_callback(user_data, result, error):
    if error:
        user_data["error"] = error
    else:
        # Record the time when the first chunk is received
        if user_data["first_chunk_time"] is None:
            user_data["first_chunk_time"] = time.time()
        # Assuming the output is a bytes array representing the audio chunk
        audio_chunk = result.as_numpy("OUT")[0]
        # Append the audio chunk to the list
        user_data["audio_chunks"].append(audio_chunk)

def main():
    # Server URL
    url = "localhost:8001"  # Adjust if your server is running elsewhere

    # Model name
    model_name = "xtts_v2"  # Ensure this matches your deployed model's name

    # Input text
    input_text = "Hello, I am Akash Sonowal"

    # Create a gRPC client
    try:
        triton_client = grpcclient.InferenceServerClient(url=url)
    except Exception as e:
        print(f"Failed to create Triton client: {e}")
        return

    # Check if the server and model are ready
    if not triton_client.is_server_live():
        print("Triton server is not live.")
        return
    if not triton_client.is_model_ready(model_name):
        print(f"Model {model_name} is not ready.")
        return

    # Prepare the input tensor
    inputs = [
        InferInput("TEXT", [1], "BYTES")
    ]
    inputs[0].set_data_from_numpy(np.array([input_text], dtype=np.object_))

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
    request_start_time = time.time()

    # Start the streaming inference
    try:
        # Start the stream with the callback function
        triton_client.start_stream(callback=partial(response_callback, user_data))

        # Send the inference request
        triton_client.async_stream_infer(model_name=model_name, inputs=inputs, outputs=outputs)

        # Signal the end of requests
        triton_client.stop_stream()

        # Check for errors
        if user_data["error"]:
            print(f"Inference failed: {user_data['error']}")
            return

    except grpcclient.InferenceServerException as e:
        print(f"Inference failed: {e}")
        return

    # Record the end time of the response
    response_end_time = time.time()

    # Calculate Time to First Chunk (TTFC)
    if user_data["first_chunk_time"]:
        ttfc = user_data["first_chunk_time"] - request_start_time
        print(f"Time to First Chunk (TTFC): {ttfc:.4f} seconds")
    else:
        print("No audio chunks received.")
        return

    # Combine all audio chunks
    complete_audio = b''.join(user_data["audio_chunks"])

    # Calculate the duration of the generated audio in seconds
    # Assuming 16-bit (2 bytes) samples and a sample rate of 22050 Hz
    audio_duration = len(complete_audio) / (2 * 22050)

    # Calculate the total processing time
    total_processing_time = response_end_time - request_start_time

    # Calculate Real-Time Factor (RTF)
    rtf = total_processing_time / audio_duration
    print(f"Real-Time Factor (RTF): {rtf:.4f}")

    # Write the complete audio to a WAV file
    with wave.open("output_audio.wav", "wb") as wf:
        # Set parameters: nchannels, sampwidth, framerate, nframes
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(22050)  # Sample rate of 22050 Hz
        wf.writeframes(complete_audio)

if __name__ == "__main__":
    main()