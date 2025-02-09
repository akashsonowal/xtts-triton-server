import os
import json
import torch
import numpy as np
import triton_python_backend_utils as pb_utils

from xtts_v2 import XTTSVocalizer
from gcs_bucket import download_all_files_in_folder

class TritonPythonModel:
    """
    Triton model for XTTS text-to-speech synthesis with streaming.
    """

    def initialize(self, args):
        """
        Model initialization - loads the XTTS model.
        """
        self.model_config = json.loads(args["model_config"]) # config.pbtxt

        # Ensure decoupled transaction policy is enabled
        if not pb_utils.using_decoupled_model_transaction_policy(self.model_config):
            raise pb_utils.TritonModelException(
                "This model requires Triton's decoupled transaction policy for streaming."
            )

        gcs_model_path = os.getenv("MODEL_PATH", "gs://swiss-knife/org_ag/vocalizer/xttsv2_mixed")
        current_directory = "/opt/tritonserver/model_repository/xtts_v2"
        self.checkpoint_dir = os.path.join(current_directory, "1", "xtts_artifacts")
        model_weights = "model.pth"
        speaker_reference_file = "clipped_first_15_seconds.wav"

        # Download model files if not present
        if not os.path.exists(self.checkpoint_dir):
            download_all_files_in_folder(gcs_model_path, self.checkpoint_dir)

        # Load the XTTS model
        self.xtts_model = XTTSVocalizer(self.checkpoint_dir, model_weights, speaker_reference_file)
        self.xtts_model.load_model(use_deepspeed=torch.cuda.is_available())

        if not self.xtts_model.is_loaded():
            raise pb_utils.TritonModelException("Failed to load XTTS model.")
        
        print("Initialized...")

    def execute(self, requests):
        """
        Processes incoming requests for text-to-speech synthesis in streaming mode.
        """

        for request in requests:
            in_input = pb_utils.get_input_tensor_by_name(request, "TEXT")
            print("in_put", in_input)
            text_data = in_input.as_numpy()[0].decode("utf-8")
            print("text_data", text_data)
            print(type(text_data))

            if not text_data.strip():
                raise pb_utils.TritonModelException("Input text is empty.")

            response_sender = request.get_response_sender()
            print("response_sender", response_sender)

            try:
                for audio_chunk in self.xtts_model.predict(text_data, "en"):
                    # out_output = pb_utils.Tensor("OUT", np.array([audio_chunk], dtype=np.object_))
                    out_output = pb_utils.Tensor("OUT", np.array([audio_chunk], dtype=np.bytes_))
                    response = pb_utils.InferenceResponse(output_tensors=[out_output])
                    response_sender.send(response)

                response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

            except Exception as e:
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                )
                response_sender.send(error_response)
                response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            
            return None
    
    def finalize(self):
        """
        `finalize` is called only once when the model is being unloaded.
        """
        print('Cleaning up...')