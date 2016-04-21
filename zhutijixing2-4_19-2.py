#coding=utf8

#print ", ".join(seg_list)
import MySQLdb
import sys
import jieba
import os
import re
import xlrd
import xlwt
import jieba.posseg as pseg

reload(sys)
sys.setdefaultencoding('utf8')


data = xlrd.open_workbook('zhutijixing.xls')

seg_list = jieba.cut("他来到了网易杭研大厦".decode("utf-8"))

sql=''
output = xlwt.Workbook()
tables=[]

tables.append(output.add_sheet("biaozhu",cell_overwrite_ok=True))
tables.append(output.add_sheet("tiqu",cell_overwrite_ok=True))
tables[0].write(0,0,'biaozhu')
tables[1].write(0,0,'tiqu')
count=0
row=1
volumn=1
cell=''
ls=[]
table = data.sheets()[0]
nrows = table.nrows
ncols = table.ncols
flag=[]
kzb=0
pair=0
for i in range(1,nrows):
	pair=0
	kzb=0
	volumn=2
	cell=' '
	ls=[]
	flag=[]
	while(volumn< ncols and cell!=''):
		cell = table.cell(i,volumn).value
		if volumn % 2 == 1:
			ls.append(cell)
			flag.append(0)
			
		print cell
		volumn+=1
	print len(ls),len(flag)
	for k in range(len(ls)):
		if (ls[k]=='n' or ls[k]=='nz' or ls[k]=='vn' or ls[k]=='ng' or ls[k]=='nl') and k<len(ls)-1 and len(table.cell(i,2+(k)*2).value)>1 and flag[k]==0:
		#""""""
			if k<len(ls)-3 and ls[k+1]=='u' and (ls[k+2]=='n' or ls[k+2]=='nz' or ls[k+2]=='vn' or ls[k+2]=='ng' or ls[k+2]=='nl' or ls[k+2]=='nr') and (ls[k+3]=='a' or ls[k+3]=='an' or ls[k+3]=='ag' or ls[k+3]=='al') and flag[k+2]==0:
				if len(table.cell(i,2+(k+2)*2).value)>1:
					print 'model 1 '+table.cell(i,2+k*2).value + ' ' + table.cell(i,2+(k+1)*2).value + ' ' + table.cell(i,2+(k+2)*2).value			
					tables[0].write(i,pair*2+1,table.cell(i,2+(k)*2).value+'的'+table.cell(i,2+(k+2)*2).value)
					tables[0].write(i,pair*2+2,table.cell(i,2+(k+3)*2).value)
					
					flag[k]=1
					flag[k+2]=1
					pair+=1
					kzb=1
			elif k<len(ls)-2 and (ls[k+1]=='n' or ls[k+1]=='nz' or ls[k+1]=='vn' or ls[k+1]=='ng' or ls[k+1]=='nl') and (ls[k+2]=='a' or ls[k+2]=='an' or ls[k+2]=='ag' or ls[k+2]=='al') and flag[k+1]==0:
			#""""""
				if len(table.cell(i,2+(k+1)*2).value)>1:
					print 'model 1 '+table.cell(i,2+k*2).value + ' ' + table.cell(i,2+(k+1)*2).value + ' ' + table.cell(i,2+(k+2)*2).value			
					tables[0].write(i,pair*2+1,table.cell(i,2+(k)*2).value+table.cell(i,2+(k+1)*2).value)
					tables[0].write(i,pair*2+2,table.cell(i,2+(k+2)*2).value)
					
					flag[k]=1
					flag[k+1]=1
					flag[k+2]=1
					pair+=1
					kzb=1
					
			elif (ls[k+1]=='a' or ls[k+1]=='an' or ls[k+1]=='ag' or ls[k+1]=='al'):
			#""""""
				if len(table.cell(i,2+(k)*2).value)>1 :
					print 'model 1 '+table.cell(i,2+k*2).value + ' ' + table.cell(i,2+(k+1)*2).value
					tables[0].write(i,pair*2+1,table.cell(i,2+(k)*2).value)
					tables[0].write(i,pair*2+2,table.cell(i,2+(k+1)*2).value)
					flag[k]=1
					flag[k+1]=1
					pair+=1
					kzb=1
		if (ls[k]=='n' or ls[k]=='nz' or ls[k]=='vn' or ls[k]=='ng' or ls[k]=='nl' or ls[k]=='nr') and k<len(ls)-2 and flag[k]==0 :
		#""""""
			if (ls[k+1]=='ad' or ls[k+1]=='d' or ls[k+1]=='vd') and (ls[k+2]=='a' or ls[k+2]=='an' or ls[k+2]=='ag' or ls[k+2]=='al'):
				if len(table.cell(i,2+(k)*2).value)>1 :
					print 'model 2 '+table.cell(i,2+k*2).value + ' ' + table.cell(i,2+(k+1)*2).value + ' ' + table.cell(i,2+(k+2)*2).value			
					tables[0].write(i,pair*2+1,table.cell(i,2+(k)*2).value)
					tables[0].write(i,pair*2+2,table.cell(i,2+(k+2)*2).value)
					flag[k]=1
					flag[k+1]=1
					flag[k+2]=1
					pair+=1				
					kzb=1
		if (ls[k]=='n' or ls[k]=='nz' or ls[k]=='vn' or ls[k]=='ng' or ls[k]=='nl') and k<len(ls)-3 and flag[k]==0 :
		#""""""
			if (ls[k+1]=='ad' or ls[k+1]=='d' or ls[k+1]=='vd') and (ls[k+2]=='ad' or ls[k+2]=='d' or ls[k+2]=='vd') and ls[k+3]=='a':
				if len(table.cell(i,2+(k+1)*2).value)>1:
					print 'model 3 '+table.cell(i,2+k*2).value + ' ' + table.cell(i,2+(k+1)*2).value + ' ' + table.cell(i,2+(k+2)*2).value + table.cell(i,2+(k+3)*2).value
					tables[0].write(i,pair*2+1,table.cell(i,2+(k)*2).value)
					tables[0].write(i,pair*2+2,table.cell(i,2+(k+3)*2).value)
					flag[k]=1
					flag[k+1]=1
					flag[k+2]=1
					flag[k+3]=1
					pair+=1
					kzb=1
		if (ls[k]=='a' or ls[k]=='an' or ls[k]=='ag' or ls[k]=='al') and k<len(ls)-1 and flag[k]==0 and flag[k+1]==0:
		#""""""
			if k<len(ls)-2 and ((ls[k+1]=='ude1' or ls[k+1]=='ude2' or ls[k+1]=='ude3') and (ls[k+2]=='n' or ls[k+2]=='vn' or ls[k+2]=='nz' or ls[k]=='ng' or ls[k]=='nl')) and flag[k+2]==0:
			#""""""
				if  len(table.cell(i,2+(k)*2).value)>1:
					print 'model 4 '+table.cell(i,2+k*2).value + ' ' + table.cell(i,2+(k+1)*2).value + ' ' + table.cell(i,2+(k+2)*2).value
					tables[0].write(i,pair*2+1,table.cell(i,2+(k+2)*2).value)
					tables[0].write(i,pair*2+2,table.cell(i,2+(k)*2).value)
					flag[k]=1
					flag[k+1]=1
					flag[k+2]=1
					pair+=1
					kzb=1
			elif ls[k+1]=='n' or ls[k+1]=='vn' or ls[k+1]=='nz' or ls[k]=='ng' or ls[k]=='nl':
				if len(table.cell(i,2+(k)*2).value)>1:				
					print 'model 4 '+table.cell(i,2+k*2).value + ' ' + table.cell(i,2+(k+1)*2).value
					tables[0].write(i,pair*2+1,table.cell(i,2+(k+1)*2).value)
					tables[0].write(i,pair*2+2,table.cell(i,2+(k)*2).value)
					flag[k]=1
					flag[k+1]=1
					pair+=1
					kzb=1
	if kzb==0:
		for k in range(len(ls)):
			if (ls[k]=='a' or ls[k]=='an' or ls[k]=='ag' or ls[k]=='al') and len(table.cell(i,2+(k)*2).value)>1:
				kzb==2
				break
	tables[0].write(i,0,kzb)
output.save('zhutijixing2.xls')
