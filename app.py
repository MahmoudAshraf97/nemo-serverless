from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
     # Create a manifest for input with below format. 
    # {'audio_filepath': /path/to/audio_file, 'offset': 0, 'duration':None, 'label': 'infer', 'text': '-', 
    # 'num_speakers': None, 'rttm_filepath': /path/to/rttm/file, 'uem_filepath'='/path/to/uem/filepath'}

    ROOT = os.getcwd()
    data_dir = os.path.join(ROOT,'data')
    os.makedirs(data_dir, exist_ok=True)

    meta = {
        'audio_filepath': 'mono_file.wav', 
        'offset': 0, 
        'duration':None, 
        'label': 'infer', 
        'text': '-',  
        'rttm_filepath': None, 
        'uem_filepath' : None
    }
    with open('data/input_manifest.json','w') as fp:
        json.dump(meta,fp)
        fp.write('\n')
        
    MODEL_CONFIG = os.path.join(data_dir,'diar_infer_telephonic.yaml')
    if not os.path.exists(MODEL_CONFIG):
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
        MODEL_CONFIG = wget.download(config_url,data_dir)

    config = OmegaConf.load(MODEL_CONFIG)
    
    pretrained_vad = 'vad_multilingual_marblenet'
    pretrained_speaker_model = 'titanet_large'
    
    config.num_workers = 1 # Workaround for multiprocessing hanging with ipython issue 

    output_dir = os.path.join(ROOT, 'outputs')
    os.makedirs(output_dir,exist_ok=True)
    config.diarizer.manifest_filepath = 'data/input_manifest.json'
    config.diarizer.out_dir = output_dir #Directory to store intermediate files and prediction outputs

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = False # compute VAD provided with model_path to vad config
    config.diarizer.clustering.parameters.oracle_num_speakers=False

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = 'diar_msdd_telephonic' # Telephonic speaker diarization model 
    
    model = NeuralDiarizer(cfg=config)
    

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    url = model_inputs.get('link', None)
    if url == None:
        return {'message': "No link provided"}
    download_target = url.split('/')[-1]
    # Download and Prepare the file
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        
        while True:
            buffer = source.read(8192)
            if not buffer:
                break
            output.write(buffer)
    
    signal, sample_rate = librosa.load(download_target, sr=None)
    soundfile.write('mono_file.wav', signal, sample_rate, 'PCM_16')
    os.remove(download_target)
    
    # Run the model
    result = model.diarize()

    # Return the results as a dictionary
    return result
