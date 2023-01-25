# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import pipeline
pretrained_vad = 'vad_multilingual_marblenet'
pretrained_speaker_model = 'titanet_large'
def download_model():
    EncDecClassificationModel.from_pretrained(model_name=pretrained_vad) #NeMo VAD
    EncDecSpeakerLabelModel.from_pretrained(model_name=pretrained_speaker_model) #NeMo Embedding

if __name__ == "__main__":
    download_model()
