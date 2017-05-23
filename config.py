'''for cluster'''
'''
    if restore:
        with main cluster:
            put model to ../../result/model_to_restore/
            set if_restore_model to True
        if continou log:
            set log_dir_global
    modify cluster_current only
    run main cluster first
    then run other

'''
status = ""
log_dir_global = "ff15-offline-feild-gamma-099-finaldiscount-4-train_on_trainingset-mix_exp-update_step_5-task_36-worker_1"
cluster_current = 0 # specific current cluster here
if_restore_model = True
model_to_restore = "../../result/model_to_restore/model.ckpt-8496809"

cluster_host = ['192.168.226.27', '192.168.226.139'] # main cluster has to be first
cluster_name = ['server'        , 'worker'] # main cluster has to be first
cluster_home = ['s'             , 'irc207'] # main cluster has to be first
cluster_main = 0 #donot modify

num_games_global = 36
games_start_global = 0
num_workers_global = 1
num_workers_total_global = num_games_global * num_workers_global
update_step = 5
