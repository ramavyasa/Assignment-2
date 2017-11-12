f = open("/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_test.txt",'r')

g = open("new_data_test",'w')
classes = set()
for i in f.readlines():
	data = i.split(' ')
	if len(data) > 2:
		keys = data[0].split(',')
		for k in keys:
			classes.add(k)

f = open("/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_test.txt",'r')
l = list(classes)
print(len(l))
for i in f.readlines():
	data = i.split(' ')
	if len(data) > 2:
		#print(' '.join(data))
		keys = data[0].split(',')
		#print(keys)
		for k in keys:
			g.write(str(l.index(k)))
			g.write(' ')
			g.write(str(' '.join(data[1:])))
			#g.write('\n')

