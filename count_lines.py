count = 0

file_dirs = './pretrained_data.txt'
filename = open(file_dirs, 'r')
file_contents = filename.read()
for file_content in file_contents:
    if file_content == '\n':
        count += 1
if file_contents[-1] != '\n':
    count += 1
print(file_dirs, count)