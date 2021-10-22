import sys

def tree(acts,words,pos):
	btree = []
	openidx = []
	wid = 0

	previous_act = 'N'

	size_tree = 0
	max_size_tree = len(words)

	for act in acts:
		if act[0] == 'S' and act[1] == 'H':
			if len(words) != 0:
				btree.append("("+pos[0]+" "+words[0]+")")
				del words[0]
				del pos[0]
				wid += 1
			previous_act = 'S'
			size_tree += 1

		elif act[0] == 'N':
			btree.insert(-1,"("+act[3:-1])
			openidx.append(len(btree)-2)
			previous_act = 'N'
		else:#REDUCE

			if len(openidx)>0:
				tmp = " ".join(btree[openidx[-1]:])+")"
				btree = btree[:openidx[-1]]
				btree.append(tmp)
				openidx = openidx[:-1]
			previous_act = 'R'


	if len(openidx)>0:
		tope = len(openidx)
		for i in range(tope):
			tmp = " ".join(btree[openidx[-1]:])+")"
			btree = btree[:openidx[-1]]
			btree.append(tmp)
			openidx = openidx[:-1]

	if len(btree)>1:
		print('(TOP', end='')
		for i in range(len(btree)):
				print(btree[i], end='')
		print(')')	
	else:
		print(btree[0])


def reorder_text(actions,text):
	stack = []
	buffer = []
	for t in reversed(range(len(text))):
		buffer.append(text[t])

	for a in actions:
		if a[1]=='H': stack.append(buffer.pop())
		if a[1]=='W': buffer.append(stack.pop(-2))

	return stack


if __name__ == "__main__":
	allpos = []
	pos = []
	for line in open(sys.argv[3]):
		line=line.strip()
		if line != 	"":
			ws = line.split("\t")
			for i in range(len(ws)):
				pos.append(ws[i])
			allpos.append(pos)
			pos = []

	text = []
	words = []
	for line in open(sys.argv[2]):
		line=line.strip()
		if line != 	"":
			ws = line.split("\t")
			for i in range(len(ws)):
                                if ws[i]=='ROOT':
                                        continue
                                words.append(ws[i])
			text.append(words)
			words = []
	allactions = []
	actions = []
	allreorder = []
	reorder = []

	sent = 0
	for line in open(sys.argv[1]):
		line=line.strip()
		if line != "":
			trans = line.split("\t")
			num_shift = 0
			num_nt = 0
			num_reduce = 0
                        
			for i in range(len(trans)):
				if trans[i][0] == 'S':
					reorder.append(trans[i])
					if trans[i][1] == 'H':
						actions.append(trans[i])
						num_shift=num_shift+1
					if trans[i][1] == 'W':
						actions.pop()
						num_shift=num_shift-1
				else:
					actions.append(trans[i])
					#if trans[i][0] == 'S': num_shift=num_shift+1
					if trans[i][0] == 'N': num_nt=num_nt+1
					if trans[i][0] == 'R': num_reduce=num_reduce+1
			
			if len(text[sent])!=num_shift:
				print('FALLO: ',sent,len(text[sent]),num_shift)
				print(text[sent])
				print(actions)
				
				exit(0)
			
			#print(reorder)
			




			#print(actions)
			allreorder.append(reorder)       
			allactions.append(actions)
			actions=[]
			reorder=[]
			sent = sent + 1

	#print(allactions[0])
	#print(text[0])
	#print(len(allactions),len(text))
	#exit(0)	

	reordered=[]
	for i in range(len(text)):
		ri=reorder_text(allreorder[i],text[i])
		reordered.append(ri)
		#print(text[i])
		#print(ri)


	reorderedpos=[]
	for i in range(len(allpos)):
		ri=reorder_text(allreorder[i],allpos[i])
		reorderedpos.append(ri)


	for i in range(len(text)):
		tree(allactions[i], reordered[i], reorderedpos[i]);
		#exit(0)
		
		
