db_root_path='./data/data_synthesis_split/bird/new_dev/dev_databases/'
data_mode='new_dev'
diff_json_path='./data/data_synthesis_split/bird/new_dev/dev.json'
# predicted_sql_path=$1
ground_truth_path='./data/data_synthesis_split/bird/new_dev/'
num_cpus=16
meta_time_out=30.0
mode_gt='gt'
mode_predict='gpt'
echo '''predicted_sql_path: $1'''
echo '''starting to compare with knowledge for ex'''
python3 -u ./bird_evaluation/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path $1 --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

echo '''starting to compare with knowledge for ves'''
python3 -u ./bird_evaluation/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path $1 --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}