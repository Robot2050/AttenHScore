nohup python -m pipeline.generate --model Meta-Llama-3-8B-Instruct --dataset coqa --fraction_of_data_to_use 1 --project_ind 0 >> nohup_llama3_8B.log 2>&1 &
# nohup python -m pipeline.generate --model Llama-2-13b-chat-hf --dataset coqa --fraction_of_data_to_use 1 --project_ind 0 >> nohup_llama2_13B.log 2>&1 &
# nohup python -m pipeline.generate --model vicuna-7b-v1.5 --dataset coqa --fraction_of_data_to_use 1 --project_ind 0 >> nohup_vicuna_7B.log 2>&1 &


# #avg or last_token
# nohup python -m pipeline.generate --model Meta-Llama-3-8B-Instruct --dataset SQuAD --fraction_of_data_to_use 1 --project_ind 1 >> nohup_llama3_8B_SQuAD_1.log 2>&1 &
# nohup python -m pipeline.generate --model Meta-Llama-3-8B-Instruct --dataset SQuAD --fraction_of_data_to_use 1 --project_ind 2 >> nohup_llama3_8B_SQuAD_2.log 2>&1 &



