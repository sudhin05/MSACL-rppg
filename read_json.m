function str = read_json(json_file_path)
fid = fopen(json_file_path);
raw = fread(fid);
str = char(raw');
fclose(fid);
str = jsondecode(str);
end
