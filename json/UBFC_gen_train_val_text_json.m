clear;
addpath('/utils'); % Ensure this directory exists and is correct

winLength = 150;
destFold_image = '/home/uas-dtu/Pai_rPPG/JAMSNet_v2/JAMSNet/Generated_Sahil';
destFold_label = '/home/uas-dtu/Pai_rPPG/JAMSNet_v2/JAMSNet/Generated_Sahil';

save_file_name = '/home/uas-dtu/Pai_rPPG/JAMSNet_v2/JAMSNet/json/UBFC_train_val_test_idx_list_Sahil.mat';
load(save_file_name); % sub_idx_train, sub_idx_val, sub_idx_test
nSub_train = length(sub_idx_train);
nSub_val = length(sub_idx_val);
nSub_test = length(sub_idx_test);

savejson = ['/home/uas-dtu/Pai_rPPG/JAMSNet_v2/JAMSNet/SavesSahil' num2str(winLength) '/json_1'];
if ~exist(savejson, 'dir') 
    mkdir(savejson);
end

% Write train JSON files
saveFile_image = [savejson '/train_image.json'];
saveFile_label = [savejson '/train_label.json'];

jsonText_image = {};
jsonText_label = {};
jsonText_cnt = 1;
for subIdx = 1:nSub_train
    iSub = sub_idx_train(subIdx);   
    subID = ['subject' num2str(iSub, '%d')];
    vidDir = [destFold_label '/' subID];
    if ~exist(vidDir, 'file')
        continue;
    end
    nMat = 0.5 * (length(dir(vidDir)) - 2);
    for iMat = 1:nMat
        imageFile = [destFold_image '/' subID '/' num2str(iMat - 1, '%04d') '.mat'];
        %labelFile = [destFold_label '/' subID '/gtPPG' num2str(iMat - 1, '%04d') '.mat'];
        jsonText_image{jsonText_cnt} = imageFile;
        %jsonText_label{jsonText_cnt} = labelFile;
        jsonText_cnt = jsonText_cnt + 1;
    end
end    
writeJSONfile(saveFile_image, jsonText_image);
%writeJSONfile(saveFile_label, jsonText_label);

% Write validation JSON files
saveFile_image = [savejson '/val_image.json'];
saveFile_label = [savejson '/val_label.json'];
jsonText_image = {};
jsonText_label = {};
jsonText_cnt = 1;
for subIdx = 1:nSub_val
    iSub = sub_idx_val(subIdx);   
    subID = ['subject' num2str(iSub, '%d')];
    vidDir = [destFold_label '/' subID];
    if ~exist(vidDir, 'file')
        continue;
    end
    nMat = 0.5 * (length(dir(vidDir)) - 2);
    for iMat = 1:nMat
        imageFile = [destFold_image '/' subID '/' num2str(iMat - 1, '%04d') '.mat'];
        labelFile = [destFold_label '/' subID '/gtPPG' num2str(iMat - 1, '%04d') '.mat'];
        jsonText_image{jsonText_cnt} = imageFile;
        jsonText_label{jsonText_cnt} = labelFile;
        jsonText_cnt = jsonText_cnt + 1;
    end
end    
writeJSONfile(saveFile_image, jsonText_image);
writeJSONfile(saveFile_label, jsonText_label);

% Write test JSON files
saveFile_image = [savejson '/test_image.json'];
saveFile_label = [savejson '/test_label.json'];
jsonText_image = {};
jsonText_label = {};
jsonText_cnt = 1;
for subIdx = 1:nSub_test
    iSub = sub_idx_test(subIdx);   
    subID = ['subject' num2str(iSub, '%d')];
    vidDir = [destFold_label '/' subID];
    if ~exist(vidDir, 'file')
        continue;
    end
    nMat = 0.5 * (length(dir(vidDir)) - 2);
    for iMat = 1:nMat
        imageFile = [destFold_image '/' subID '/' num2str(iMat - 1, '%04d') '.mat'];
        labelFile = [destFold_label '/' subID '/gtPPG' num2str(iMat - 1, '%04d') '.mat'];
        jsonText_image{jsonText_cnt} = imageFile;
        jsonText_label{jsonText_cnt} = labelFile;
        jsonText_cnt = jsonText_cnt + 1;
    end
end    
writeJSONfile(saveFile_image, jsonText_image);
writeJSONfile(saveFile_label, jsonText_label);

% Define writeJSONfile function if it doesn't already exist
function writeJSONfile(filename, data)
    jsonText = jsonencode(data);
    fid = fopen(filename, 'w');
    if fid == -1
        error('Cannot create JSON file: %s', filename);
    end
    fwrite(fid, jsonText, 'char');
    fclose(fid);
end
