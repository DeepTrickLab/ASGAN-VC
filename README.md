# Autoformer

## Speaker Embedding

Train your own Speaker Embedding [here](https://github.com/licaiwang/d-vector) or use my [pre-trained model](https://drive.google.com/file/d/1-KY9H9JAiZwhi3xoJjmxJ4hVIBE4mG4p/view?usp=sharing) with LstmDV

The pre-trained model is trained by 913 speaker with 53 utterances , Download the dataset from [openSLR train-clean-360.tar.gz](https://www.openslr.org/12) and ignore the speaker wich utterances number is lower than 50, the model performance is test with 40 speaker from VCTK dataset.

| Model | LstmDV | MetaDV |
| ----- | ------ | ------ |
| EER   | 3%     | 2%     |

## Data Prepare

- Put your Speaker Embedding model in ./model/static/model.pt
- Run make_spec.ipynb and make_metadata.ipynb with the data as following format.

       - model
         - static
           - model.pt
       - make_data
          - factory
          - wavs
              - 225 (include many audio data)
              - 226
              - ...
              - ...
          - make_metadata.ipynb
          - make_spec.ipynb

- After that you will get a ./spmel (default name) folder and a train.pkl, copy ./spmel to root dir.

## Training

### Available Model

- AutoVC  (Original Implement)
- AutoVC2 (Original Implement + Adain)
- AutoVC3 (Original Implement + Modulated Conv)
- MetaVC  (Use MLPMixer and Metaformer)
- MetaVC2 (Use MLPMixer and Metaformer + Adain)
- MetaVC3 (Use MLPMixer and Metaformer + Modulated Conv)

### For Original Training

    python train.py --model_name=AutoVC --data_dir=spmel --save_model_name=model_name

### For Training with GAN

    python train_with_gan.py --model_name=AutoVC --data_dir=spmel --save_model_name=model_name

### For Training with StarGan

    python train_with_stargan.py --model_name=AutoVC --data_dir=spmel --save_model_name=model_name
