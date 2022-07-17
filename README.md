# ASGAN-VC

## Speaker Embedding 

### For VC training

Train your own Speaker Embedding [here](https://github.com/licaiwang/metadv) or use my pre-trained model [here](https://drive.google.com/file/d/1nF-nq4vb3PGOFp04iN2IC8jKVFeGCE5I/view?usp=sharing) with MetaDV

### For VC Evaluate

The pre-trained model is trained by 913 speaker with 53 utterances , Download the dataset from [openSLR train-clean-360.tar.gz](https://www.openslr.org/12) and ignore the speaker wich utterances number is lower than 50, the model performance is test with 80 speaker and each speaker has 100 utterence, we random select 16 utterence for making real emdding and remain 84 for evaluate; you can download our pre-trained model [here](https://drive.google.com/file/d/1WfJOhK0vFHKlZXZ66by142efYlDxq3jW/view?usp=sharing)

| Model | LstmDV | MetaDV |
| ----- | ------ | ------ |
| EER   | 8.68%  | 5.75%  |
| AUC   | 97.05% | 98.81% |

### For VC Evaluate with thirdparty

We use deep-speaker with their ResCNN Softmax+Triplet pre-trained model for Evaluate, althought it is an Unofficial Tensorflow/Keras implementation, but it reproduce the performance as the [paper](https://arxiv.org/pdf/1705.02304.pdf) claim, for more detail please check in their [repo](https://github.com/philipperemy/deep-speaker).


## Vocoder 
 
We use MelGan pre-trained model(multi_speaker.pt) to generate waveform from mel, for more detail please check their [official repo](https://github.com/descriptinc/melgan-neurips)


## VC Model

### Data Prepare

- Put your Speaker Embedding model in ./model/static/model.pt
- Run make_spec.ipynb and make_metadata.ipynb with the data as following format.

       - model
         - static
           - model.pt
       - wavs
            - 225 (include many audio data)
            - 226
            - ...
            - ...
        - make_metadata.ipynb
        - make_spec.ipynb

- After that you will get a ./spmel (default name) folder with a train.pkl inside.

### Training

#### Available Model

- AutoVC  
- ASGANVC 
- ASGANVC_AdaIN

#### Available Taining Method

- Original
- GAN
- BiGAN
- SNGAN

#### Training Config (static)

- check the json file in ./config

#### Training Config (dynamic)
      
    --model_name:str, VC structure from factory
    --use_config:str, VC static config in ./config
    --save_model_name: str, name of saving model after training 
    --data_dir:str, your training dir path 
    --method:str, training method from trainer,defalut is Original
    --device:str, defalut is cuda:0
    --is_adain:bool, is this VC structure has adain, defalut is False
    --is_validate: bool, validate while training, defalut is False

#### Example

    python train.py --model_name=AutoVC --save_model_name=model_name --data_dir=spmel

