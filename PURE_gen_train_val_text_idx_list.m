clear;
nSub = 10;
train_test_ratio = 0.8;
train_val_ratio = 0.8;
save_file_name = 'PURE_train_val_test_idx_list_1.mat';
%% train val test split
idx = randperm(nSub);
nSub_train = round(train_val_ratio*train_test_ratio*nSub);
nSub_test = nSub-round(train_test_ratio*nSub);
nSub_val = nSub - nSub_train - nSub_test;
sub_idx_train = idx(1:nSub_train);
sub_idx_val = idx(nSub_train+1:nSub_train+nSub_val);
sub_idx_test = idx(end-nSub_test+1:end);
save(save_file_name,'sub_idx_train','sub_idx_val','sub_idx_test');
