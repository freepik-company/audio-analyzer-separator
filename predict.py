# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from typing import List, Optional
from cog import BasePredictor, BaseModel, Input, Path

import torch
import allin1
import os 

# ----------------------- AUDIO-SEPARATOR , DEIXEI OS EXECUTANDO PARALELAMENTE------------------------
# import torch
print("TORCH GPU CUDA is available: ", torch.cuda.is_available())

# CHECK IF ONNXGPU IS AVAILABLE
import onnxruntime
print("ONNXRUNTIME TYPE: ", onnxruntime.get_device())

# import audio_separator.utils.cli
from audio_separator.separator import Separator
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

class Output(BaseModel):    
    analyzer_result: Optional[Path]
    visualization: Optional[Path]
    sonification: Optional[Path]
    mdx_vocals: Optional[Path]
    mdx_instrumental: Optional[Path]
    mdx_other: Optional[List[Path]]
    demucs_vocals: Optional[Path]
    demucs_bass: Optional[Path]
    demucs_drums: Optional[Path]
    demucs_guitar: Optional[Path]
    demucs_piano: Optional[Path]
    demucs_other: Optional[Path]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(
        self,
        music_input: Path = Input(
            description="An audio file input to analyze.",
            default=None,
        ),
        visualize: bool = Input(
            description="Save visualizations",
            default=False,
        ),
        sonify: bool = Input(
            description="Save sonifications",
            default=False,
        ),
        # activ: bool = Input(
        #     description="Save frame-level raw activations from sigmoid and softmax",
        #     default=False,
        # ),
        # embed: bool = Input(
        #     description="Save frame-level embeddings",
        #     default=False,
        # ),
        model: str = Input(
            description="Name of the pretrained model to use",
            default="harmonix-all",
            choices=["harmonix-all", "harmonix-fold0", "harmonix-fold1", "harmonix-fold2", "harmonix-fold3", "harmonix-fold4", "harmonix-fold5", "harmonix-fold6", "harmonix-fold7"]
        ),
        include_activations: bool = Input(
            description="Whether to include activations in the analysis results or not.",
            default=False
        ),
        include_embeddings: bool = Input(
            description="Whether to include embeddings in the analysis results or not.",
            default=False,
        ),
        audioSeparator: bool = Input(
            description="Separate the audio into vocals and instrumental with MDX-net (best for music)",
            default=False
        ),
        audioSeparatorModel: str = Input(
            description="Name of the pretrained model to use: https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json",
            default="Kim_Vocal_2.onnx",
        ),
    ) ->  Output:
        
        output_dir = {}
       
        # copy all files inside ./static_models/ to /tmp/audio-separator-models/. create if not exists
        if not os.path.exists('/tmp/audio-separator-models/'):
            os.makedirs('/tmp/audio-separator-models/')
        os.system("cp -r ./static_models/* /tmp/audio-separator-models/")

        if not music_input:
            raise ValueError("Must provide `music_input`.")

        # Clean up directories if they exist
        for dir_name in ['demix', 'spec', 'output', 'viz', 'sonif']:
            import shutil
            if os.path.isdir(dir_name):
                shutil.rmtree(dir_name)

        # Music Structure Analysis
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []

            if audioSeparator:
                futures.append(executor.submit(self.run_separator, audioSeparatorModel, music_input))

            futures.append(executor.submit(self.run_allin1_analyze, music_input, visualize, sonify, model, include_activations, include_embeddings))

            for future in as_completed(futures):
                output_dir.update(future.result())

        return Output(
            analyzer_result=output_dir.get("analyzer_result"),
            visualization=output_dir.get("visualization"),
            sonification=output_dir.get("sonification"),
            mdx_vocals=output_dir.get("mdx_vocals"),
            mdx_instrumental=output_dir.get("mdx_instrumental"),
            mdx_other=output_dir.get("mdx_other"),
            demucs_vocals=output_dir.get("demucs_vocals"),
            demucs_bass=output_dir.get("demucs_bass"),
            demucs_drums=output_dir.get("demucs_drums"),
            demucs_guitar=output_dir.get("demucs_guitar"),
            demucs_piano=output_dir.get("demucs_piano"),
            demucs_other=output_dir.get("demucs_other")            
        )
            # return output_dir
    
    def run_separator(self, model_name, music_input):
        separator_output_dir = {}

        # PARAMS: https://github.com/karaokenerds/python-audio-separator?tab=readme-ov-file#parameters-for-the-separator-class
        separator = Separator(log_level=logging.INFO)

        # (if unspecified, defaults to 'UVR-MDX-NET-Inst_HQ_3.onnx')
        separator.load_model(model_name)
        stem_output_paths = separator.separate(music_input)

        separator_output_dir["mdx_other"] = []

        for stem_output_path in stem_output_paths:
            print(f'Stem saved at {stem_output_path}')
            # if includes Instrumental add to separator_output_dir["mdx_instrumental"] if includes Vocals add to separator_output_dir["mdx_vocals"]
            if "Instrumental" in str(stem_output_path):
                separator_output_dir["mdx_instrumental"] = Path(stem_output_path)
            if "Vocals" in str(stem_output_path):
                separator_output_dir["mdx_vocals"] = Path(stem_output_path)
            # if not "Instrumental" or "Vocals" add to separator_output_dir["mdx_other"]
            if "Instrumental" not in str(stem_output_path) and "Vocals" not in str(stem_output_path):
                separator_output_dir["mdx_other"].append(Path(stem_output_path) )

        return separator_output_dir

    def run_allin1_analyze(self, music_input, visualize, sonify, model, include_activations, include_embeddings):
        allin1_output_dir = {}

        allin1.analyze(paths=music_input, out_dir='output', visualize=visualize, sonify=sonify, model=model, device=self.device, include_activations=include_activations, include_embeddings=include_embeddings, keep_byproducts=True, )
        
        music_input_name = str(music_input).rsplit('/', 1)[-1].rsplit('.', 1)[0]

        # add to output_dir all files in demix/htdemucs/music_input_name if folder exists
        if os.path.isdir('demix/htdemucs/'+music_input_name):
            for dirpath, dirnames, filenames in os.walk('demix/htdemucs/'+music_input_name):
                for filename in [f for f in filenames if f.rsplit('.', 1)[-1] == "wav"]:
                    demix_dir = os.path.join(dirpath, filename)
                    print("Will add to output: ", demix_dir)
                    # get only name without file extension
                    allin1_output_dir["demucs_"+filename.split('.')[0]] = Path(demix_dir)
            

        for dirpath, dirnames, filenames in os.walk("output"):
            for filename in [f for f in filenames if f.rsplit('.', 1)[-1] == "json"]:
                json_dir = os.path.join(dirpath, filename)
        allin1_output_dir["analyzer_result"]=Path(json_dir)

        if visualize:
            for dirpath, dirnames, filenames in os.walk("viz"):
                for filename in [f for f in filenames if f.rsplit('.', 1)[-1] == "pdf"]:
                    visualization_dir = os.path.join(dirpath, filename)
                    import fitz
                    doc = fitz.open(str(visualization_dir))
                    for i, page in enumerate(doc):
                        img = page.get_pixmap()
                        img_dir = str(visualization_dir).rsplit('.',1)[0]+'.png'
                        img.save(img_dir)
                        break
            allin1_output_dir["visualization"]=Path(img_dir)

        if sonify:
            for dirpath, dirnames, filenames in os.walk("sonif"):
                for filename in [f for f in filenames if f.rsplit('.', 1)[-1] == "mp3"]:
                    sonification_dir = os.path.join(dirpath, filename)
            allin1_output_dir["sonification"]=Path(sonification_dir)

        return allin1_output_dir