# -*- coding=utf8 -*-
import json
import sys
import jieba
import os
import re
import xlrd
import xlwt
import jieba.posseg as pseg
import copy, collections

reload(sys)
sys.setdefaultencoding('utf8')

def get_ciduibiao(ciduibiao):
	file1 = open('cidui.txt')
	text = file1.read()
	L=text.split('\n')
	for l in L:
		if len(l.split('\t'))>1:
			ciduibiao.append([l.split('\t')[0],l.split('\t')[1].split('\r')[0]])
	file1.close()

def get_modelTable(modelTable):
	data = xlrd.open_workbook('trainA.xls')
	table = data.sheets()[0]
	nrows = table.nrows
	ncols = table.ncols
	data2 = xlrd.open_workbook('trainB.xls')
	table2 = data2.sheets()[0]
	nrows2 = table2.nrows
	ncols2 = table2.ncols
	row=1
	
	while row<nrows :
		col2=0
		while(col2<ncols2-1):
			biaozhu1=table2.cell(row,col2+1).value
			biaozhu2=table2.cell(row,col2+2).value
			if biaozhu1=='':
				break
			if biaozhu1!='':
				flag=0
				additem=''
				col1=1
				while(1):
					if col1<ncols:
						fenci=table.cell(row,col1).value
						cixing=table.cell(row,col1+1).value
					else:
						fenci=''
					if flag==0 and (fenci in biaozhu1 or biaozhu1 in fenci) and cixing!='':
						flag=1
						additem+=cixing
						#print '1', biaozhu1, fenci
						#print 'newbegin'
					elif flag==1 and (fenci in biaozhu2 or biaozhu2 in fenci) and cixing!='':
						flag=2
						additem+=','
						additem+=cixing
						#print '2', biaozhu2, fenci
						#print 'newend'
					elif flag==1 and cixing!='':
						additem+=','
						additem+=cixing
						#print '1', biaozhu2, fenci
						#print 'new continue'
					elif flag==2:
						break
					elif fenci=='':
						break
					col1+=2
					#print additem
				#print 'end'
				if flag==2 and (additem in modelTable):
					modelTable[additem]+=1
				elif flag==2 and 'a' in additem:
					modelTable[additem]=1
			col2+=2
		row+=1
		#print '-------------------------------------'
		#print 'rowOK'
	for item in modelTable:
		#print item,modelTable[item]
                pass

def fenci(Num, segbook, typenum, segtables, sourcename):
	data = xlrd.open_workbook(sourcename)
	table = data.sheets()[typenum]
	nrows = table.nrows
	ncols = table.ncols
	row=1
	col=1
	ls=[]
	lsw=[]
	if ((Num%8)%4)%2==1:
		jieba.load_userdict('userdict.txt')
	while row<nrows :
		col=1
		length=0
		cell = table.cell(row,col).value
		s=cell
		ls=[]
		lsw=[]
		seg_list = pseg.cut(s.decode("utf-8"))
		for w in seg_list:
			length+=1
		if length<50:
			if ((Num%8)%4)/2==0:
				seg_list = pseg.cut(s.decode("utf-8"))
				for w in seg_list:
					segtables[typenum].write(row,col,w.word)
					col+=1
					segtables[typenum].write(row,col,w.flag)
					col+=1
					
			else:
				cell = table.cell(row,col).value
				s=cell
				seg_list = pseg.cut(s.decode("utf-8"))
				for ww in seg_list:
					ls.append(ww.flag)
					lsw.append(ww.word)
				for i in range(length):
					if i-1>0:
						if ls[i-1]=='uj' and ls[i]!='n':
							ls[i]='n'
					if i-1>0 and i-2>=0:
						if (ls[i-2]=='n' or ls[i-2]=='nz' or ls[i-2]=='vn' or ls[i-2]=='ng' or ls[i-2]=='nl') and (ls[i-1]=='d' or ls[i-1]=='vd' or ls[i-1]=='ad' or ls[i-1]=='zg') and ls[i]!='a':
							ls[i]='a'
				for j in range(length):
					segtables[typenum].write(row,col,lsw[j])
					col+=1
					segtables[typenum].write(row,col,ls[j])
					col+=1
		row+=1	
	segbook.save('Segmentation.xls')

def tiqu(Num, capbook, segbook, typenum, captables, segtables, ciduibiao, sourcename):
	fenci(Num, segbook, typenum, segtables, sourcename)
	
	data = xlrd.open_workbook('Segmentation.xls')
	table = data.sheets()[typenum]
	nrows = table.nrows
	ncols = table.ncols

	data2 = xlrd.open_workbook(sourcename)
	table2 = data2.sheets()[typenum]

	for i in range(1,nrows):
		kzb=0
		cell=' '
		cell2=' '
		ls=[]
		flag=[]
		col=1
		pair=0
		equal=0
		while(col< ncols and cell!=''):
			cell = table.cell(i,col).value
			if col % 2 == 0:
				ls.append(cell)
				flag.append(0)
			col+=1
		if Num>=8:
			for model in modelTable:
				mm=model.split(',')
				c=0
				while(c<len(ls)):
					equal=0
					if ls[c]==mm[0]:
						equal=1
						zhutici=''
						jixingci=''
						for ii in range(len(mm)):
							if mm[ii]!=ls[c]:
								equal=0
								break
							if ls[c]=='n' or ls[c]=='nz' or ls[c]=='vn' or ls[c]=='ng' or ls[c]=='nl':
								zhutici+=table.cell(i,c*2+1).value
							if ls[c]=='a' or ls[c]=='an' or ls[c]=='ag' or ls[c]=='al':
								jixingci+=table.cell(i,c*2+2).value
					c+=1				
				if equal==1:
					captables[typenum].write(i,pair*2+1,zhutici)
					captables[typenum].write(i,pair*2+2,jixingci)
					pair+=1
					kzb=1
			
		for k in range(len(ls)):
			if (ls[k]=='n' or ls[k]=='nz' or ls[k]=='vn' or ls[k]=='ng' or ls[k]=='nl') and k<len(ls)-1 and len(table.cell(i,1+(k)*2).value)>1 and flag[k]==0:
				if k<len(ls)-4 and ls[k+1]=='uj' and (ls[k+2]=='n' or ls[k+2]=='nz' or ls[k+2]=='vn' or ls[k+2]=='ng' or ls[k+2]=='nl' or ls[k+2]=='nr') and (ls[k+3]=='ad' or ls[k+3]=='d' or ls[k+3]=='vd' or ls[k+3]=='zg') and (ls[k+4]=='a' or ls[k+4]=='an' or ls[k+4]=='ag' or ls[k+4]=='al') and flag[k+2]==0:
					if len(table.cell(i,1+(k+2)*2).value)>1:
						#print 'model 1 '+table.cell(i,1+k*2).value + ' ' + table.cell(i,1+(k+1)*2).value + ' ' + table.cell(i,2+(k+2)*2).value			
						captables[typenum].write(i,pair*2+1,table.cell(i,1+(k)*2).value+'的'+table.cell(i,1+(k+2)*2).value)
						captables[typenum].write(i,pair*2+2,table.cell(i,1+(k+4)*2).value)
						if (Num%8)/4==1:
							if not( [table.cell(i,1+(k)*2).value+'的'+table.cell(i,1+(k+2)*2).value,table.cell(i,1+(k+4)*2).value] in	ciduibiao):
								ciduibiao.append( [table.cell(i,1+(k)*2).value+'的'+table.cell(i,1+(k+2)*2).value,table.cell(i,1+(k+4)*2).value])			
								#print 'update ciduibiao'
						flag[k]=1
						flag[k+2]=1
						pair+=1
						kzb=1
				elif k<len(ls)-3 and ls[k+1]=='uj' and (ls[k+2]=='n' or ls[k+2]=='nz' or ls[k+2]=='vn' or ls[k+2]=='ng' or ls[k+2]=='nl' or ls[k+2]=='nr') and (ls[k+3]=='a' or ls[k+3]=='an' or ls[k+3]=='ag' or ls[k+3]=='al') and flag[k+2]==0:
					if len(table.cell(i,1+(k+2)*2).value)>1:
						#print 'model 1 '+table.cell(i,1+k*2).value + ' ' + table.cell(i,1+(k+1)*2).value + ' ' + table.cell(i,1+(k+2)*2).value			
						captables[typenum].write(i,pair*2+1,table.cell(i,1+(k)*2).value+'的'+table.cell(i,1+(k+2)*2).value)
						captables[typenum].write(i,pair*2+2,table.cell(i,1+(k+3)*2).value)
						if (Num%8)/4==1:
							if not( [table.cell(i,1+(k)*2).value+'的'+table.cell(i,1+(k+2)*2).value,table.cell(i,1+(k+3)*2).value] in	ciduibiao):
								ciduibiao.append( [table.cell(i,1+(k)*2).value+'的'+table.cell(i,1+(k+2)*2).value,table.cell(i,1+(k+3)*2).value] )			
								#print 'update ciduibiao'
						flag[k]=1
						flag[k+2]=1
						pair+=1
						kzb=1
				elif k<len(ls)-2 and (ls[k+1]=='n' or ls[k+1]=='nz' or ls[k+1]=='vn' or ls[k+1]=='ng' or ls[k+1]=='nl') and (ls[k+2]=='a' or ls[k+2]=='an' or ls[k+2]=='ag' or ls[k+2]=='al') and flag[k+1]==0:
				#""""""
					if len(table.cell(i,1+(k+1)*2).value)>1:
						#print 'model 1 '+table.cell(i,1+k*2).value + ' ' + table.cell(i,1+(k+1)*2).value + ' ' + table.cell(i,1+(k+2)*2).value			
						captables[typenum].write(i,pair*2+1,table.cell(i,1+(k)*2).value+table.cell(i,1+(k+1)*2).value)
						captables[typenum].write(i,pair*2+2,table.cell(i,1+(k+2)*2).value)
						
						if (Num%8)/4==1:
							if not( [table.cell(i,1+(k)*2).value+table.cell(i,1+(k+1)*2).value,table.cell(i,1+(k+2)*2).value] in ciduibiao):
								ciduibiao.append( [table.cell(i,1+(k)*2).value+table.cell(i,1+(k+1)*2).value,table.cell(i,1+(k+2)*2).value] )			
								#print 'update ciduibiao'
						flag[k]=1
						flag[k+1]=1
						flag[k+2]=1
						pair+=1
						kzb=1
					
				elif (ls[k+1]=='a' or ls[k+1]=='an' or ls[k+1]=='ag' or ls[k+1]=='al'):
				#""""""
					if len(table.cell(i,1+(k)*2).value)>1 :
						#print 'model 1 '+table.cell(i,1+k*2).value + ' ' + table.cell(i,1+(k+1)*2).value
						captables[typenum].write(i,pair*2+1,table.cell(i,1+(k)*2).value)
						captables[typenum].write(i,pair*2+2,table.cell(i,1+(k+1)*2).value)

						if (Num%8)/4==1:
							if not( [table.cell(i,1+(k)*2).value,table.cell(i,1+(k+1)*2).value] in ciduibiao):
								ciduibiao.append( [table.cell(i,1+(k)*2).value,table.cell(i,1+(k+1)*2).value] )			
								#print 'update ciduibiao'

						flag[k]=1
						flag[k+1]=1
						pair+=1
						kzb=1
			if (ls[k]=='n' or ls[k]=='nz' or ls[k]=='vn' or ls[k]=='ng' or ls[k]=='nl' or ls[k]=='nr') and k<len(ls)-2 and flag[k]==0 :
			#""""""
				if (ls[k+1]=='ad' or ls[k+1]=='d' or ls[k+1]=='vd' or ls[k+1]=='zg') and (ls[k+2]=='a' or ls[k+2]=='an' or ls[k+2]=='ag' or ls[k+2]=='al'):
					if len(table.cell(i,1+(k)*2).value)>1 :
						#print 'model 2 '+table.cell(i,1+k*2).value + ' ' + table.cell(i,1+(k+1)*2).value + ' ' + table.cell(i,1+(k+2)*2).value			
						captables[typenum].write(i,pair*2+1,table.cell(i,1+(k)*2).value)
						captables[typenum].write(i,pair*2+2,table.cell(i,1+(k+2)*2).value)

						if (Num%8)/4==1:
							if not( [table.cell(i,1+(k)*2).value,table.cell(i,1+(k+2)*2).value] in ciduibiao):
								ciduibiao.append( [table.cell(i,1+(k)*2).value,table.cell(i,1+(k+2)*2).value] )			
								#print 'update ciduibiao'

						flag[k]=1
						flag[k+1]=1
						flag[k+2]=1
						pair+=1				
						kzb=1
			if (ls[k]=='n' or ls[k]=='nz' or ls[k]=='vn' or ls[k]=='ng' or ls[k]=='nl') and k<len(ls)-3 and flag[k]==0 :
			#""""""
				if (ls[k+1]=='ad' or ls[k+1]=='d' or ls[k+1]=='vd' or ls[k+1]=='zg') and (ls[k+2]=='ad' or ls[k+2]=='d' or ls[k+2]=='vd') and ls[k+3]=='a':
					if len(table.cell(i,1+(k+1)*2).value)>1:
						#print 'model 3 '+table.cell(i,1+k*2).value + ' ' + table.cell(i,1+(k+1)*2).value + ' ' + table.cell(i,1+(k+2)*2).value + table.cell(i,1+(k+3)*2).value
						captables[typenum].write(i,pair*2+1,table.cell(i,1+(k)*2).value)
						captables[typenum].write(i,pair*2+2,table.cell(i,1+(k+3)*2).value)

						if (Num%8)/4==1:
							if not( [table.cell(i,1+(k)*2).value,table.cell(i,1+(k+3)*2).value] in ciduibiao):
								ciduibiao.append( [table.cell(i,1+(k)*2).value,table.cell(i,1+(k+3)*2).value] )			
								#print 'update ciduibiao'

						flag[k]=1
						flag[k+1]=1
						flag[k+2]=1
						flag[k+3]=1
						pair+=1
						kzb=1
			if (ls[k]=='a' or ls[k]=='an' or ls[k]=='ag' or ls[k]=='al') and k<len(ls)-1 and flag[k]==0 and flag[k+1]==0:
			#""""""
				if k<len(ls)-2 and ((ls[k+1]=='uj') and (ls[k+2]=='n' or ls[k+2]=='vn' or ls[k+2]=='nz' or ls[k]=='ng' or ls[k]=='nl')) and flag[k+2]==0:
				#""""""
					if  len(table.cell(i,1+(k)*2).value)>1:
						#print 'model 4 '+table.cell(i,1+k*2).value + ' ' + table.cell(i,1+(k+1)*2).value + ' ' + table.cell(i,1+(k+2)*2).value
						captables[typenum].write(i,pair*2+1,table.cell(i,1+(k+2)*2).value)
						captables[typenum].write(i,pair*2+2,table.cell(i,1+(k)*2).value)

						if (Num%8)/4==1:
							if not( [table.cell(i,1+(k+2)*2).value,table.cell(i,1+(k)*2).value] in ciduibiao):
								ciduibiao.append( [table.cell(i,1+(k+2)*2).value,table.cell(i,1+(k)*2).value] )			
	
						flag[k]=1
						flag[k+1]=1
						flag[k+2]=1
						pair+=1
						kzb=1
				elif ls[k+1]=='n' or ls[k+1]=='vn' or ls[k+1]=='nz' or ls[k]=='ng' or ls[k]=='nl':
					if len(table.cell(i,1+(k)*2).value)>1:				
						captables[typenum].write(i,pair*2+1,table.cell(i,1+(k+1)*2).value)
						captables[typenum].write(i,pair*2+2,table.cell(i,1+(k)*2).value)

						if (Num%8)/4==1:
							if not( [table.cell(i,1+(k+1)*2).value,table.cell(i,1+(k)*2).value] in ciduibiao):
								ciduibiao.append( [table.cell(i,1+(k+1)*2).value,table.cell(i,1+(k)*2).value] )			
						flag[k]=1
						flag[k+1]=1
						pair+=1
						kzb=1
		if (Num%8)/4==1:
			cell2 = table2.cell(i,1).value
			for cd in ciduibiao:
                                pass
	capbook.save('Extraction.xls')

def quchong(typenum, extractname, finalbook, finaltables):
        data = xlrd.open_workbook(extractname)
        table = data.sheets()[typenum]
        nrows = table.nrows
        ncols = table.ncols
        currow = 1
        flag = 0
        topiclists = []
        adjlists = []
        freqlists = []
        while(currow < nrows):
                curcol = 1
                while curcol + 1 <= ncols:
                        topic = table.cell(currow, curcol).value
                        adj = table.cell(currow, curcol + 1).value
                        if topic == '' or adj == '':
                                curcol += 2
                                continue
                        flag = 0
                        for j in range(len(topiclists)):
                                if topic in topiclists[j] and (adj in adjlists[j] or adjlists[j] in adj):
                                        topiclists[j] = topic
                                        freqlists[j] += 1
                                        if adj in adjlists[j]:
                                                adjlists[j] = adj
                                        flag = 1
                                        break
                                elif topiclists[j] in topic and (adj in adjlists[j] or adjlists[j] in adj):
                                        freqlists[j] += 1
                                        if adj in adjlists[j]:
                                                adjlists[j] = adj
                                        flag = 1
                                        break
                        if flag == 0:
                                topiclists.append(topic)
                                adjlists.append(adj)
                                freqlists.append(1)  
                        curcol += 2
                currow += 1
        for i in range(len(topiclists)):
                finaltables[typenum].write(i + 1, 1, topiclists[i])
                finaltables[typenum].write(i + 1, 2, adjlists[i])
                finaltables[typenum].write(i + 1, 3, freqlists[i])
        finalbook.save('FinalResult.xls')

if __name__=="__main__":
        Types = {}
        fin = file('testset.json')
        workbook = xlwt.Workbook()
        segbook = xlwt.Workbook()
        capbook = xlwt.Workbook()
        finalbook = xlwt.Workbook()
        tables = []
        segtables = []
        captables = []
        finaltables = []
        ciduibiao=[]
        modelTable=collections.OrderedDict()
        s = json.load(fin)
        for i in range(len(s)):
                item = s[i]
                if item[2] in Types:
                        Types[item[2]].append(item[4])
                else:
                        Types[item[2]] = []
                        Types[item[2]].append(item[4])
        for i in range(len(Types.keys())):
                tables.append(workbook.add_sheet(Types.keys()[i], cell_overwrite_ok = True))
                tables[i].write(0, 0, Types.keys()[i])
                for j in range(len(Types[Types.keys()[i]])):
                        tables[i].write(j + 1, 1, Types[Types.keys()[i]][j])
        fin.close()
        workbook.save('TestSet.xls')
        for i in range(len(Types.keys())):
                segtables.append(segbook.add_sheet(Types.keys()[i], cell_overwrite_ok = True))
                segtables[i].write(0, 0, Types.keys()[i])
                captables.append(capbook.add_sheet(Types.keys()[i], cell_overwrite_ok = True))
                captables[i].write(0, 0, Types.keys()[i])
                tiqu(13, capbook, segbook, i, captables, segtables, ciduibiao, u'TestSet.xls')
        for i in range(len(Types.keys())):
                finaltables.append(finalbook.add_sheet(Types.keys()[i], cell_overwrite_ok = True))
                finaltables[i].write(0, 0, Types.keys()[i])
                quchong(i, u'Extraction.xls', finalbook, finaltables)
        
        
