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


data = xlrd.open_workbook('RPF_raw.xls')

seg_list = jieba.cut("他来到了网易杭研大厦".decode("utf-8"))

sql=''
#output = xlwt.Workbook()
#tables=[]

#tables.append(output.add_sheet("biaozhu",cell_overwrite_ok=True))
#tables.append(output.add_sheet("tiqu",cell_overwrite_ok=True))
#tables[0].write(0,0,'biaozhu')
#tables[1].write(0,0,'tiqu')
count=0
row=1
volumn=1
cell=''
ls=[]
table0 = data.sheets()[0]
table1 = data.sheets()[1]
nrows = table0.nrows
ncols = table0.ncols
print nrows,ncols
flag=[]
kzb=0

M=0
N=0
O=0
R=0
P=0
F=0
for i in range(1,nrows):
	
	volumn=1
	cell1=' '
	
	while(volumn< ncols and cell1!=''):
		cell1=str(table1.cell(i,volumn).value)
		cell0=str(table0.cell(i,volumn).value)
		
		
		if volumn%2==1 and cell1!='':
			M+=1
			if cell0 !='':
				N+=1
				if (table1.cell(i,volumn+1).value in table0.cell(i,volumn+1).value or table0.cell(i,volumn+1).value in table1.cell(i,volumn+1).value or table1.cell(i,volumn+1).value==table0.cell(i,volumn+1).value) and (cell1 in cell0 or cell0 in cell1 or cell1==cell0):
				#table1.cell(i,volumn+1).value in table0.cell(i,volumn+1).value or table0.cell(i,volumn+1).value in table1.cell(i,volumn+1).value or     
				#cell1 in cell0 or cell0 in cell1 or
					O+=1
		volumn+=1
		print i,volumn

R=(O+0.0)/M

P=(O+0.0)/N

F=2*R*P/(R+P)
print "多" in "太多"
#F=2*P*R/(P+R)
print M,N,O,R,P,F

	
#output.save('RPF.xls')
