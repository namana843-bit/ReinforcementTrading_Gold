@echo off
cd /d C:\Users\Naman\ReinforcementTrading_Part_1
python train_agent.py --csv_path data/gold_data_10y.csv --timesteps 150000 --results_dir ./results > train2_log.txt 2>&1
