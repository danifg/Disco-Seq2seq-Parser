import sys

filepath = sys.argv[1]
f = open(filepath)


for line in f:
    line = line.strip()
    #print(line)

    if len(line)==0:


    else:
        size_sent+=1
        fields = line.split('\t')



f.close()
