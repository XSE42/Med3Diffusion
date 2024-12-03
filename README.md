# MedDiffusion

## Steps

### Envrionment

- [conda](https://docs.anaconda.com/miniconda/) is recommended

    ```
    conda create -n med3d python=3.10
    conda activate med3d

    conda install numpy
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install tqdm
    pip install accelerate transformers datasets
    pip install git+https://github.com/huggingface/diffusers
    pip install tensorboard
    conda install -c simpleitk simpleitk
    conda install pandas
    conda install imageio
    conda install -c conda-forge monai
    pip install scipy
    ```

### Dataset

- Need the metadata file `ADNI_CN_meta.json` of cognitively normal MRIs in ADNI

    ```
    {
        "scan_list": [
            {
                "subject": "011_S_0002",
                "gender": "M",
                "scan_time": "2005-08-26",
                "scan_age": 74.3
            },
            ...,
            {
                "subject": "011_S_0005",
                "gender": "M",
                "scan_time": "2006-09-05",
                "scan_age": 74.69
            }
        ],
        "sub_list": [
            "024_S_5290",
            ...,
            "005_S_0610"
        ]
    }
    ```

### Preprocess

1. Split the dataset at a ratio of `Train:Val:Test = 7:1:2`

    ```
    python src/preprocess/preprocess_adni_split.py --meta data/ADNI_meta/ADNI_CN_meta.json --output_dir data/ADNI_meta
    ```

    The output files are `data/ADNI_meta/ADNI_CN_{Train|Val|Test}_meta`

2. Split dataset into 4 age groups

    ```
    python src/preprocess/preprocess_adni_agesplit.py --meta data/ADNI_meta/ADNI_CN_Train_meta.json --output_dir data/ADNI_meta
    python src/preprocess/preprocess_adni_agesplit.py --meta data/ADNI_meta/ADNI_CN_Val_meta.json --output_dir data/ADNI_meta
    python src/preprocess/preprocess_adni_agesplit.py --meta data/ADNI_meta/ADNI_CN_Test_meta.json --output_dir data/ADNI_meta
    ```

    The output files are `data/ADNI_meta/ADNI_CN_{Train|Val|Test}_meta_age{60|70|80|90}`

3. Obtain a sampler for edge sampling during generation

    ```
    python src/preprocess/preprocess_adni_metasampler.py --meta data/ADNI_meta/ADNI_CN_Train_meta.json --outfile data/ADNI_meta/ADNI_CN_Train_sampler.json
    ```

4. Obtain metadata of the synthetic dataset

    ```
    python src/preprocess/preprocess_synthetic_meta.py --meta data/ADNI_meta/ADNI_CN_Train_meta.json --output_dir data/ADNI_meta --meta_prefix Synth_CN_Train --scan_prefix train
    ```

5. Obtain `init_atlas` (mean of MRIs in the training set)

    ```
    python src/preprocess/preprocess_init_atlas_adni.py \
      --data_dir "data/ADNI" \
      --output_dir "data/ADNI_meta/init_atlas" \
      --meta "data/ADNI_meta/ADNI_CN_Train_meta.json"
    ```

    The output files are `data/ADNI_meta/init_atlas/init_atlas.{pt|npy|nii.gz}`

6. Obtain ADNI_128 dataset with volume shape `(104, 128, 104)`

    ```
    python src/preprocess/preprocess_adni128.py \
      --data_dir "data/ADNI" \
      --output_dir "data/ADNI_128" \
      --meta "data/ADNI_meta/ADNI_CN_meta.json"
    ```

7. Obtain edge detection results of ADNI_128 dataset

    ```
    python src/preprocess/preprocess_adni128_canny.py --data_dir data/ADNI_128 --meta data/ADNI_meta/ADNI_CN_meta.json
    ```

### Training

1. Train VAE

    ```
    CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
      --num_machines=1 \
      --gpu_ids='0' \
      --mixed_precision=no \
      --dynamo_backend=no \
      src/scripts/train_vae_adni128.py \
      --data_dir="data/ADNI" \
      --meta="data/ADNI_meta/ADNI_CN_Train_meta.json" \
      --val_meta="data/ADNI_meta/ADNI_CN_Val_meta.json" \
      --output_dir="outputs/vae_adni128" \
      --data_shape 104 128 104 \
      --train_batch_size=2 \
      --val_size=8 \
      --dataloader_num_workers=4 \
      --num_epochs=300 \
      --save_volume_epochs=25 \
      --save_model_epochs=50 \
      --gradient_accumulation_steps=1 \
      --learning_rate=1e-4 \
      --lr_warmup_steps=500 \
      --kl_weight=1e-7 \
      --checkpointing_steps=8000 \
      --checkpoints_total_limit=2
    ```

2. Train 3D LDM

    ```
    CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
      --num_machines=1 \
      --gpu_ids='0' \
      --mixed_precision=no \
      --dynamo_backend=no \
      src/scripts/train_numcond_adni128.py \
      --pretrained_model="outputs/vae_adni128" \
      --data_dir="data/ADNI" \
      --meta="data/ADNI_meta/ADNI_CN_Train_meta.json" \
      --output_dir="outputs/numcond_ldm_adni128" \
      --data_shape 104 128 104 \
      --train_batch_size=2 \
      --dataloader_num_workers=4 \
      --num_train_epochs=500 \
      --gradient_accumulation_steps=1 \
      --learning_rate=1e-4 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --prediction_type="epsilon" \
      --checkpointing_steps=5000 \
      --checkpoints_total_limit=2 \
      --validation_epochs=25
    ```

3. Generate Synth_numcond_128 dataset with 3D LDM

    ```
    python src/scripts/generate_numcond_synthdata_128.py \
      --pretrained_model outputs/numcond_ldm_adni128 \
      --data_dir data/Synth_numcond_128 \
      --meta data/ADNI_meta/Synth_CN_Train.json \
      --gpu 0
    ```

4. Train Med3Diffusion

    ```
    CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
      --num_machines=1 \
      --gpu_ids='0' \
      --mixed_precision=no \
      --dynamo_backend=no \
      src/scripts/train_med3diffusion_adni128.py \
      --pretrained_model="outputs/vae_adni128" \
      --data_dir="data/ADNI_128" \
      --meta="data/ADNI_meta/ADNI_CN_Train_meta.json" \
      --val_meta="data/ADNI_meta/ADNI_CN_Val_meta.json" \
      --meta_sampler="data/ADNI_meta/ADNI_CN_Train_sampler.json" \
      --output_dir="outputs/med3diffusion_adni128" \
      --data_shape 104 128 104 \
      --train_batch_size=2 \
      --val_size=8 \
      --dataloader_num_workers=4 \
      --num_train_epochs=500 \
      --gradient_accumulation_steps=1 \
      --learning_rate=1e-4 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --prediction_type="epsilon" \
      --checkpointing_steps=5000 \
      --checkpoints_total_limit=2 \
      --validation_epochs=20
    ```

5. Generate Synth_med3diffusion_128 dataset with Med3Diffusion

    ```
    python src/scripts/generate_med3diffusion_synthdata_128.py \
      --pretrained_model outputs/med3diffusion_adni128 \
      --data_dir data/Synth_med3diffusion_128 \
      --edge_dir data/ADNI_128 \
      --meta data/ADNI_meta/Synth_CN_Train.json \
      --meta_sampler data/ADNI_meta/ADNI_CN_Train_sampler.json \
      --gpu 0
    ```

6. Train Voxelmorph

    ```
    CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
      --num_machines=1 \
      --gpu_ids='0' \
      --mixed_precision=no \
      --dynamo_backend=no \
      src/scripts/train_vxm_adni128_pad.py \
      --data_dir="data/ADNI_128" \
      --meta="data/ADNI_meta/ADNI_CN_Train_meta.json" \
      --val_meta="data/ADNI_meta/ADNI_CN_Val_meta.json" \
      --output_dir="outputs/vxm_adni128_pad" \
      --train_batch_size=8 \
      --val_size=8 \
      --dataloader_num_workers=8 \
      --num_epochs=800 \
      --save_volume_epochs=50 \
      --save_model_epochs=100 \
      --gradient_accumulation_steps=1 \
      --learning_rate=1e-4 \
      --lr_warmup_steps=500 \
      --grad_weight=0.01 \
      --checkpointing_steps=8000 \
      --checkpoints_total_limit=2
    ```

7. Train Voxelmorph for atlas construction

    ```
    CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
      --num_machines=1 \
      --gpu_ids='0' \
      --mixed_precision=no \
      --dynamo_backend=no \
      src/scripts/train_atlas_vxm_adni128_pad.py \
      --pretrained_model="outputs/vxm_adni128_pad" \
      --data_dir="data/ADNI_128" \
      --init_atlas_dir="data/ADNI_meta/init_atlas" \
      --meta="data/ADNI_meta/ADNI_CN_Train_meta.json" \
      --val_meta="data/ADNI_meta/ADNI_CN_Val_meta.json" \
      --output_dir="outputs/atlas_vxm_adni128_pad" \
      --train_batch_size=8 \
      --val_size=8 \
      --dataloader_num_workers=8 \
      --num_epochs=500 \
      --save_volume_epochs=50 \
      --save_model_epochs=100 \
      --gradient_accumulation_steps=1 \
      --learning_rate=1e-4 \
      --atlas_lr=1e-4 \
      --lr_warmup_steps=500 \
      --sim_weight=1.0 \
      --grad_weight=0.3 \
      --deform_weight=1e-2 \
      --checkpointing_steps=3200 \
      --checkpoints_total_limit=2

    CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
      --num_machines=1 \
      --gpu_ids='0' \
      --mixed_precision=no \
      --dynamo_backend=no \
      src/scripts/train_atlas_vxm_adni128_pad.py \
      --pretrained_model="outputs/vxm_adni128_pad" \
      --data_dir="data/ADNI_128" \
      --synth_data_dir="data/Synth_numcond_128" \
      --init_atlas_dir="data/ADNI_meta/init_atlas" \
      --meta="data/ADNI_meta/ADNI_CN_Train_meta.json" \
      --synth_meta="data/ADNI_meta/Synth_CN_Train.json" \
      --val_meta="data/ADNI_meta/ADNI_CN_Val_meta.json" \
      --output_dir="outputs/atlas_vxm_synnumcond_pad" \
      --train_batch_size=8 \
      --val_size=8 \
      --dataloader_num_workers=8 \
      --num_epochs=400 \
      --save_volume_epochs=50 \
      --save_model_epochs=100 \
      --gradient_accumulation_steps=1 \
      --learning_rate=1e-4 \
      --atlas_lr=1e-4 \
      --lr_warmup_steps=500 \
      --sim_weight=1.0 \
      --grad_weight=0.3 \
      --deform_weight=1e-2 \
      --checkpointing_steps=3200 \
      --checkpoints_total_limit=2

    CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
      --num_machines=1 \
      --gpu_ids='0' \
      --mixed_precision=no \
      --dynamo_backend=no \
      src/scripts/train_atlas_vxm_adni128_pad.py \
      --pretrained_model="outputs/vxm_adni128_pad" \
      --data_dir="data/ADNI_128" \
      --synth_data_dir="data/Synth_med3diffusion_128" \
      --init_atlas_dir="data/ADNI_meta/init_atlas" \
      --meta="data/ADNI_meta/ADNI_CN_Train_meta.json" \
      --synth_meta="data/ADNI_meta/Synth_CN_Train.json" \
      --val_meta="data/ADNI_meta/ADNI_CN_Val_meta.json" \
      --output_dir="outputs/atlas_vxm_synmed3diffusion_pad" \
      --train_batch_size=8 \
      --val_size=8 \
      --dataloader_num_workers=8 \
      --num_epochs=400 \
      --save_volume_epochs=50 \
      --save_model_epochs=100 \
      --gradient_accumulation_steps=1 \
      --learning_rate=1e-4 \
      --atlas_lr=1e-4 \
      --lr_warmup_steps=500 \
      --sim_weight=1.0 \
      --grad_weight=0.3 \
      --deform_weight=1e-2 \
      --checkpointing_steps=3200 \
      --checkpoints_total_limit=2
    ```

### Inference

1. Generate atlases with Voxelmorph

    ```
    python src/scripts/infer_atlas_vxm_adni128.py \
      --pretrained_model outputs/atlas_vxm_adni128_pad \
      --data_dir data/ADNI_128 \
      --init_atlas_dir data/ADNI_meta/init_atlas \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --postfix vxm_adni128 \
      --gpu 0

    python src/scripts/infer_atlas_vxm_adni128.py \
      --pretrained_model outputs/atlas_vxm_synnumcond_pad \
      --data_dir data/ADNI_128 \
      --init_atlas_dir data/ADNI_meta/init_atlas \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --postfix vxm_synnumcond \
      --gpu 0

    python src/scripts/infer_atlas_vxm_adni128.py \
      --pretrained_model outputs/atlas_vxm_synmed3diffusion_pad \
      --data_dir data/ADNI_128 \
      --init_atlas_dir data/ADNI_meta/init_atlas \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --postfix vxm_synmed3diffusion \
      --gpu 0
    ```

### Eval FID

1. Generate data under testset conditions for FID computation

    ```
    # NumCond LDM
    python src/scripts/generate_testset_numcond_synthdata_128.py \
      --pretrained_model outputs/numcond_ldm_adni128 \
      --data_dir data/ADNI_128 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --gpu 0

    # Med3Diffusion
    python src/scripts/generate_testset_med3diffusion_synthdata_128.py \
      --pretrained_model outputs/med3diffusion_adni128 \
      --data_dir data/ADNI_128 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --gpu 0
    ```

2. Extract feature of testset data with [MedicalNet](https://github.com/Tencent/MedicalNet)

    ```
    # Med3D
    export med3d_path=/path/to/Med3D_Code
    cp src/scripts/extract_feat_med3d.py $med3d_path
    cd $med3d_path
    python extract_feat_med3d.py \
      --data_dir data/ADNI_128 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --infile "t1.pt" \
      --outfile "t1_feat_med3d.pt" \
      --pretrained_model trails/models/resnet_50_epoch_110_batch_0.pth.tar \
      --depth 104 \
      --height 128 \
      --width 104 \
      --gpu 0

    python extract_feat_med3d.py \
      --data_dir data/ADNI_128 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --infile "numcond_synth.pt" \
      --outfile "numcond_synth_feat_med3d.pt" \
      --pretrained_model trails/models/resnet_50_epoch_110_batch_0.pth.tar \
      --depth 104 \
      --height 128 \
      --width 104 \
      --gpu 0

    python extract_feat_med3d.py \
      --data_dir data/ADNI_128 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --infile "med3diffusion_synth.pt" \
      --outfile "med3diffusion_synth_feat_med3d.pt" \
      --pretrained_model trails/models/resnet_50_epoch_110_batch_0.pth.tar \
      --depth 104 \
      --height 128 \
      --width 104 \
      --gpu 0
    ```

3. Compute FID

    ```
    python src/scripts/eval_fid.py \
      --data_dir data/ADNI_128 \
      --meta1 data/ADNI_meta/ADNI_CN_Test_meta.json \
      --meta2 data/ADNI_meta/ADNI_CN_Test_meta.json \
      --file1 "t1_feat_med3d.pt" \
      --file2 "numcond_synth_feat_med3d.pt" \
      --gpu 0

    python src/scripts/eval_fid.py \
      --data_dir data/ADNI_128 \
      --meta1 data/ADNI_meta/ADNI_CN_Test_meta.json \
      --meta2 data/ADNI_meta/ADNI_CN_Test_meta.json \
      --file1 "t1_feat_med3d.pt" \
      --file2 "med3diffusion_synth_feat_med3d.pt" \
      --gpu 0
    ```


### Eval Dice Score

1. Generate list of data for segmentation

    ```
    python src/scripts/generate_synthseg_list.py \
      --data_dir data/ADNI_128 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --name t1_pad \
      --infile results/testset_t1_pad_in.txt \
      --outfile results/testset_t1_pad_out.txt \
      --resamplefile results/testset_t1_pad_resample.txt

    python src/scripts/generate_synthseg_list.py \
      --data_dir data/ADNI_128 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --name atlas_vxm_adni128 \
      --infile results/testset_atlas_vxm_adni128_in.txt \
      --outfile results/testset_atlas_vxm_adni128_out.txt \
      --resamplefile results/testset_atlas_vxm_adni128_resample.txt

    python src/scripts/generate_synthseg_list.py \
      --data_dir data/ADNI_128 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --name atlas_vxm_synnumcond \
      --infile results/testset_atlas_vxm_synnumcond_in.txt \
      --outfile results/testset_atlas_vxm_synnumcond_out.txt \
      --resamplefile results/testset_atlas_vxm_synnumcond_resample.txt

    python src/scripts/generate_synthseg_list.py \
      --data_dir data/ADNI_128 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --name atlas_vxm_synmed3diffusion \
      --infile results/testset_atlas_vxm_synmed3diffusion_in.txt \
      --outfile results/testset_atlas_vxm_synmed3diffusion_out.txt \
      --resamplefile results/testset_atlas_vxm_synmed3diffusion_resample.txt

    python src/scripts/generate_synthseg_list.py \
      --data_dir data/ADNI_128 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --name numcond_synth_pad \
      --infile results/testset_numcond_synth_pad_in.txt \
      --outfile results/testset_numcond_synth_pad_out.txt \
      --resamplefile results/testset_numcond_synth_pad_resample.txt

    python src/scripts/generate_synthseg_list.py \
      --data_dir data/ADNI_128 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --name med3diffusion_synth_pad \
      --infile results/testset_med3diffusion_synth_pad_in.txt \
      --outfile results/testset_med3diffusion_synth_pad_out.txt \
      --resamplefile results/testset_med3diffusion_synth_pad_resample.txt
    ```

2. Obtain segmentation masks with [SynthSeg](https://github.com/BBillot/SynthSeg)

    ```
    conda activate synthseg_310

    CUDA_VISIBLE_DEVICES=0 python scripts/commands/SynthSeg_predict.py \
      --i ~/code/diffmodel/results/testset_t1_pad_in.txt \
      --o ~/code/diffmodel/results/testset_t1_pad_out.txt \
      --resample ~/code/diffmodel/results/testset_t1_pad_resample.txt

    CUDA_VISIBLE_DEVICES=0 python scripts/commands/SynthSeg_predict.py \
      --i ~/code/diffmodel/results/testset_atlas_vxm_adni128_in.txt \
      --o ~/code/diffmodel/results/testset_atlas_vxm_adni128_out.txt \
      --resample ~/code/diffmodel/results/testset_atlas_vxm_adni128_resample.txt

    CUDA_VISIBLE_DEVICES=0 python scripts/commands/SynthSeg_predict.py \
      --i ~/code/diffmodel/results/testset_atlas_vxm_synnumcond_in.txt \
      --o ~/code/diffmodel/results/testset_atlas_vxm_synnumcond_out.txt \
      --resample ~/code/diffmodel/results/testset_atlas_vxm_synnumcond_resample.txt

    CUDA_VISIBLE_DEVICES=0 python scripts/commands/SynthSeg_predict.py \
      --i ~/code/diffmodel/results/testset_atlas_vxm_synmed3diffusion_in.txt \
      --o ~/code/diffmodel/results/testset_atlas_vxm_synmed3diffusion_out.txt \
      --resample ~/code/diffmodel/results/testset_atlas_vxm_synmed3diffusion_resample.txt

    CUDA_VISIBLE_DEVICES=0 python scripts/commands/SynthSeg_predict.py \
      --i ~/code/diffmodel/results/testset_numcond_synth_pad_in.txt \
      --o ~/code/diffmodel/results/testset_numcond_synth_pad_out.txt \
      --resample ~/code/diffmodel/results/testset_numcond_synth_pad_resample.txt

    CUDA_VISIBLE_DEVICES=0 python scripts/commands/SynthSeg_predict.py \
      --i ~/code/diffmodel/results/testset_med3diffusion_synth_pad_in.txt \
      --o ~/code/diffmodel/results/testset_med3diffusion_synth_pad_out.txt \
      --resample ~/code/diffmodel/results/testset_med3diffusion_synth_pad_resample.txt
    ```

3. Evaluate the structural integrity

    ```
    python src/scripts/eval_structure.py \
      --pretrained_model outputs/vxm_adni128_pad \
      --data_dir data/ADNI_128 \
      --output_dir results/structure_numcond \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --pred numcond_synth_pad \
      --gt t1_pad \
      --gpu 0

    python src/scripts/eval_structure.py \
      --pretrained_model outputs/vxm_adni128_pad \
      --data_dir data/ADNI_128 \
      --output_dir results/structure_numcond_age60 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta_age60.json \
      --pred numcond_synth_pad \
      --gt t1_pad \
      --gpu 0

    python src/scripts/eval_structure.py \
      --pretrained_model outputs/vxm_adni128_pad \
      --data_dir data/ADNI_128 \
      --output_dir results/structure_med3diffusion \
      --meta data/ADNI_meta/ADNI_CN_Test_meta.json \
      --pred med3diffusion_synth_pad \
      --gt t1_pad \
      --gpu 0

    python src/scripts/eval_structure.py \
      --pretrained_model outputs/vxm_adni128_pad \
      --data_dir data/ADNI_128 \
      --output_dir results/structure_med3diffusion_age60 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta_age60.json \
      --pred med3diffusion_synth_pad \
      --gt t1_pad \
      --gpu 0
    ```

4. Compute dice scores

    ```
    python src/scripts/eval_atlas_vxm_synthseg.py \
      --pretrained_model outputs/atlas_vxm_adni128_pad \
      --data_dir data/ADNI_128 \
      --output_dir results/age60 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta_age60.json \
      --postfix vxm_adni128 \
      --gpu 0

    python src/scripts/eval_atlas_vxm_synthseg.py \
      --pretrained_model outputs/atlas_vxm_synnumcond_pad \
      --data_dir data/ADNI_128 \
      --output_dir results/age60 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta_age60.json \
      --postfix vxm_synnumcond \
      --gpu 0

    python src/scripts/eval_atlas_vxm_synthseg.py \
      --pretrained_model outputs/atlas_vxm_synmed3diffusion_pad \
      --data_dir data/ADNI_128 \
      --output_dir results/age60 \
      --meta data/ADNI_meta/ADNI_CN_Test_meta_age60.json \
      --postfix vxm_synmed3diffusion \
      --gpu 0
    ```
