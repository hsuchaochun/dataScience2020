python3 gan.py --mode=process_data --data_path=$1
# python3 gan.py --mode=train --n_epochs=2
python3 gan.py --mode=train --n_epochs=1000 --batch_size=1024 --lr=1E-5 --sample_interval=20
python3 gan.py --mode=inference --model_path=result/models/20800/

