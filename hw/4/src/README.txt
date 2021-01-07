##### Folders #####
data/ :
	1) images/ : contains raw png files (initial)
	2) annotaions/ : contains bounding boxes annotation files. (initial)
	3) images_crop/ : contains cropped images according to annotations. (after STEP 1.)
	4) images_ref/ : extract first K cropped images for FID calculation. (after STEP 1.)
eval/ : contains codes to calculate FID
results/ :
	1) images_train/ : contains sampled generated images during training, named with currently finished batches. (during STEP 2.)	
	2) models/ : contains sub-folders of trained models, named with currently finished batches. (during STEP 2.)
	3) images_inference/: contains generated images for inference (after STEP 3.)


##### STEP 0. Environment #####
pip install -r requirements.txt

The environment is based on CUDA 10.0.130 and CuDNN 7.6.3. You can change torch/torchvision versions according to your CUDA/CuDNN versions. The codes are developed with general modules in Pytorch, and it should work fine with other versions. However, you may still need to handle dependency problems in some cases.


##### STEP 1. Preprocessing ######
python gan.py --mode=process_data --data_path=data/

The data_path should be a folder containing two sub-folders: 1) images/ and 2) annotations/.


##### STEP 2. Training ######
python gan.py --mode=train

Please finish the model and loss design ([TODO] in the codes) and adjust the hyperparameters ([ADJUST] in the codes)


##### STEP 3. Inference ######
python gan.py --mode=inference --model_path=result/models/0/

The model path should be a folder containing two model files: 1) discriminator.pt and 2) generator.pt


##### STEP 4. Evaluation ######
python eval/fid_score.py result/images_inference/ data/images_ref/

The first argument is a folder containing generated images, and the second argument is a folder containing reference images.


##### Script for evaluation ######
To generate images into result/images_inference from scratch, please run "./run.sh data/" . Note that the script requires one argument data_path.


	
