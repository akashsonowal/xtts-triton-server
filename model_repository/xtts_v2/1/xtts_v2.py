import os
import torch
import numpy as np
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

from loggercz import get_logger
from base_tts import BaseVocalizer


logger = get_logger()

class XTTSVocalizer(BaseVocalizer):
    def __init__(self, checkpoint_dir, model, speaker):
        super().__init__(checkpoint_dir)
        self.model_path = self.checkpoint_dir / model
        self.speaker_path = self.checkpoint_dir / speaker  # Renamed for clarity
        self.model_loaded = False  # Initialize the model_loaded attribute to False
        self.model = None  # Initialize model as None

    def load_model(self, use_deepspeed):
        logger.info("Loading model from directory: %s", self.checkpoint_dir)

        # Check if reference audio file exists
        if not os.path.exists(self.speaker_path):
            logger.error("Reference audio file does not exist: %s", self.speaker_path)
            raise FileNotFoundError(f"Reference audio not found: {self.speaker_path}")

        try:
            config = XttsConfig()
            config.load_json(self.checkpoint_dir / "config.json")

            # Attempt to load the model with GPU support if available
            if torch.cuda.is_available():
                self.model = Xtts.init_from_config(config)
                self.model.load_checkpoint(config, checkpoint_dir=self.checkpoint_dir, eval=True, use_deepspeed=use_deepspeed)
                if use_deepspeed:
                    self.model.cuda()  # Move model to GPU
                    logger.info("Model loaded and moved to GPU.")
            else:
                logger.warning("CUDA is not available. Attempting to load model on CPU.")
                self.model = Xtts.init_from_config(config)
                self.model.load_checkpoint(config, checkpoint_dir=self.checkpoint_dir, eval=True, use_deepspeed=False)  # Disable DeepSpeed on CPU

            # Extract speaker embeddings and conditioning latents
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=[self.speaker_path])
            self.speaker = {
                "speaker_embedding": speaker_embedding,
                "gpt_cond_latent": gpt_cond_latent,
            }
            logger.info("Speaker embeddings and latent conditioning extracted.")
            self.model_loaded = True  # Set the model_loaded attribute to True

        except Exception as e:
            logger.error("Failed to load model: %s", str(e))
            raise

    def is_loaded(self):
        return self.model_loaded

    def predict(self, text, language="en"):
        logger.info("Generating audio for text: '%s' in language: '%s'", text, language)
        if not self.is_loaded():
            logger.error("Model is not loaded. Cannot perform prediction.")
            raise RuntimeError("Model is not loaded.")

        try:
            streamer = self.model.inference_stream(
                text,
                language,
                self.speaker["gpt_cond_latent"],
                self.speaker["speaker_embedding"],
                speed=0.85,
                temperature=0.8,
            )
            for chunk in streamer:
                processed_bytes = (chunk.cpu().numpy() * 32767).astype(np.int16).tobytes()
                yield processed_bytes
        except Exception as e:
            logger.error("Error during prediction: %s", str(e))
            raise

if __name__ == "__main__":
    try:
        xtts_vocalizer = XTTSVocalizer("/home/ubuntu/akashsonowal/mockingjay/src/artifacts/xtts_v2", "model.pth", "clipped_first_15_seconds.wav")
        xtts_vocalizer.load_model()
        print(xtts_vocalizer.is_loaded())  # Should print True if the model is loaded successfully
    except Exception as e:
        logger.error("Initialization error: %s", str(e))