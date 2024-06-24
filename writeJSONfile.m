function writeJSONfile(saveFile,jsonText)
fileID = fopen(saveFile,'w');
jsonText = jsonencode(jsonText);
fprintf(fileID,'%s',jsonText);
fclose(fileID);
end
