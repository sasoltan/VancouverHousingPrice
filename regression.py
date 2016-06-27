from pyspark import SparkConf, SparkContext,SQLContext
from pyspark.mllib.common import  _py2java, _java2py
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LinearRegressionModel
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.feature import StandardScalerModel
from pyspark.mllib.feature import PCA 
from pyspark.mllib.linalg import Vectors
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql import DataFrameNaFunctions
from pyspark.sql.types import *
import sys, operator
from ast import literal_eval

# isEmpty = udf(lambda x: len(x) == 0, BooleanType())
def string2Float(df):
	df = df.select(df.PID, df.legalType.cast(FloatType()).alias('legalType'), df.zoneCat.cast(FloatType()).alias('zoneCat'),\
				   df.FSA.cast(FloatType()).alias('FSA'), df.curVal.cast(FloatType()).alias('curVal'), df.curImpVal.cast(FloatType()).alias('curImpVal'),\
	               df.preVal.cast(FloatType()).alias('preVal'), df.preImpVal.cast(FloatType()).alias('preImpVal'),\
	               df.year.cast(FloatType()).alias('year'), df.interestrates.cast(FloatType()).alias('interestrates'),\
	               df.oilprice.cast(FloatType()).alias('oilprice'), df.conversionrate.cast(FloatType()).alias('conversionrate'))
	
	df = df.filter((df['curVal'].isNotNull()) & (df['curImpVal'].isNotNull()) & (df['preVal'].isNotNull()) & (df['preImpVal'].isNotNull()) & (df['year'].isNotNull()) &\
				   (df['oilprice'].isNotNull()) & (df['conversionrate'].isNotNull()) & (df['interestrates'].isNotNull()) &\
				   (df['legalType'].isNotNull()) & (df['zoneCat'].isNotNull()) & (df['FSA'].isNotNull()))               
	return df


def findCorrelation(df, cols):
	n = len(cols)
	Matrix = [0 for x in range(n)]
	for j in range (0, n): 
		Matrix[j] = df.corr(cols[0], cols[j])
	return Matrix


def setWeight(df, cols, corr):
	n = len(cols)
	for i in range(0,n): 
		df.withColumn(cols[i], corr[i]*df[cols[i]])
	return df

def findClass(value):
	if (value >= 1600000) :
		return '6'
	if (value >= 1400000) :
		return '5'
	if (value >= 1000000) :
		return '4'
	if (value >= 700000) :
		return '3'
	if (value >= 300000) :
		return '2'
	else :
		return '1'

def makeCode(i, len):
    code = '0' * len
    code = code[0:i] + '1' + code[i+1:]
    return code

def getCodes(n):
	encoding = []
	for i in range(n) :
     		encoding.append(makeCode(i, n))
	return encoding
		    
# input is like "11101,201389,2006,...."
# output is like [1,1,1,0,1,201389,2006,...]
def data2vec(dataStr):
	ls = dataStr.split(',')
	if (len(ls[0]) != 41) :
		 return [] 
	l = map(lambda x:x, ls[0]) 
	l = l + ls[1:]
	#print " THE ERROR IS "+str(l)
	result = [float(x) for x in l]	
	return result
def hotEncoding(colname, length, df): 
	codes = getCodes(length)
	df.withColumn(colname, str(df[colname]))
	df.replace([str(x) for x in range(3)], codes ,colname)
	return df

def createDataset(df) :
	df = hotEncoding ('legalType', 3, df)
	df = hotEncoding ('zoneCat', 9, df)
	df = hotEncoding ('FSA', 29, df)
	df = df.withColumn('LegalZoneFSA', concat(df.zoneCat, df.legalType, df.FSA)).cache()
	df.rdd.map(lambda x : str(x.year) + ' ,' + str(x.LegalZoneFSA) + ' ,'+ str(x.zoneCat) + ' ,' + str(x.FSA) + ' ,'+ str(x.legalType)).coalesce(1).saveAsTextFile(myPath)
	df = df.select(df.PID, df.LegalZoneFSA, df.curVal, df.curImpVal, df.preVal, df.preImpVal, df.year, df.interestrates, df.oilprice, df.conversionrate)
	cols = ['curImpVal', 'year', 'preVal', 'preImpVal' ,'interestrates', 'oilprice', 'conversionrate']
	des = df.describe(*cols)    
	for col in cols :
		list = des.select(des[col]).map(lambda x: float(x[0])).collect()    # returns (count, mean, stddev, min, max) for each col
		if (col == 'year') :
			p1 = list[3]
			p2 = list[4] - list[3] + 5
		else :      
			p1 = list[1]
			p2 = list[2]
		df = df.withColumn(col, (df[col] - p1) / p2)
	cols = ['LegalZoneFSA','curImpVal', 'year', 'preVal', 'preImpVal', 'oilprice', 'interestrates', 'conversionrate']
	getData = udf(data2vec, ArrayType(FloatType()))
	df = df.withColumn('vector', getData(concat_ws(',', *cols)))
	df.registerTempTable("temp")
	#result = sqlContext.sql("SELECT PID, curVal, vector FROM temp WHERE size(vector) = 48")
	# print result.count()
	return result

def PCAdata(df, num): 
	Label = df.map(lambda p: p.label).zipWithIndex().map(lambda (label, index): (index, label))
	Features = df.map(lambda p: p.features)
	pcaModel = PCA(num).fit(Features)
	projected = pcaModel.transform(Features)
	second = projected.zipWithIndex().map(lambda (features, index): (index, features))
	result = Label.join(second).map(lambda (idx, (label, features)) : LabeledPoint(label, features))
	return result
	
#inputs = sys.argv[1]
#output = sys.argv[2]
inputs = "/Users/saeeds/Desktop/housingprices/Datasets/ETLData"
output = "/users/saeeds/Desktop/housingprices/Datasets/ETLData"
myPath = "/users/saeeds/Desktop/data"

conf = SparkConf().setAppName('housingprice')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
# Load and parse the data
# each row of data is considered as one sample so that the first element is label and the second is a vector of features
# line format: (PID, curVal, legalType, zoneCat, FSA, curImpVal, year, preVal, preImpVal, oilPrice, interestRate, conversionRate)
df = sqlContext.read.parquet(inputs + "/training")
df.show()
'''
df = string2Float(df)
cols = ['curVal', 'curImpVal', 'preVal', 'preImpVal', 'year', 'interestrates', 'oilprice', 'conversionrate', 'zoneCat', 'legalType', 'FSA']
corr = findCorrelation(df, cols)
# df.rdd.map(lambda x : str(x.year) + ' ,' +  str(x.zoneCat) + ' ,' + str(x.FSA) + ' ,'+ str(x.legalType)).coalesce(1).saveAsTextFile(myPath)
weightedcols = ['curVal', 'curImpVal', 'preVal', 'preImpVal', 'year', 'interestrates', 'oilprice', 'conversionrate']
df = setWeight(df, weightedcols, corr)
df = createDataset(df)
setClass = udf(findClass, StringType())
df = df.withColumn('ClassNum', setClass(df.curVal)).rdd.cache()
trainData = df.map(lambda (PID, label, ClassNum, features): LabeledPoint(ClassNum, features)) 

df2 = sqlContext.read.parquet(inputs + "/validation")
df2 = string2Float(df2)
df2 = setWeight(df2, weightedcols, corr)
df2 = createDataset(df2).cache()
df2 = df2.withColumn('ClassNum', setClass(df2.curVal)).rdd.cache()
PID = df2.zipWithIndex().map(lambda ((PID, label, ClassNum, features), index):(index, PID, label, ClassNum, features)).cache()
validData = df2.map(lambda (PID, label, ClassNum, features): LabeledPoint(ClassNum, features))#.cache()

# featureNum =40
# trainData = PCAdata(trainData, featureNum).cache()
# validData = PCAdata(validData, featureNum).cache()

logesticmodel = LogisticRegressionWithLBFGS.train(trainData, iterations = 100, intercept=True, numClasses=6)
vps = validData.map(lambda p: (p.label, model.predict(p.features))).cache()
MSE = (vps.filter(lambda (a,b): a==b)).count()/vps.count()

# create a lineaner regression in each calss
vps = vps.map(lambda (v, p): p).zipWithIndex().map(lambda (p,idx): (idx,p)).join(PID).map(lambda(indx, (p, (PID, label, ClassNum, features))): (p, PID, label, features))
vps.write.parquet(output+"/LogResult")
dfc = sqlContext.read.parquet(output+"/LogResult")
dfc.registerTempTable("dataClass")
f = open(output+"/results", 'w')  
for i in range(1,7):
	classx = sqlContext.sql("SELECT PID, label, features FROM  dataClass WHERE p="+str(i))
	trainx, testx = classx.randomSplit([0.7, 0.3], seed=0)
	trainx = trainx.rdd.map(lambda (PID, label, features): LabeledPoint(label, features))
	model = LinearRegressionWithSGD.train(trainx, regType = "l2" , regParam =0)
	vpsx = testx.map(lambda p: (p.label, model.predict(p.features))).cache()
	MSEx = (vpsx.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / vpsx.count())**0.5
	f.write(("Testing class " + str(i) + ": " +str(MSEx) + "\n"))
f.close()	
'''
# setenv SPARK_HOME /Volumes/bshadgar/Spark/spark-1.5.2-bin-hadoop2.6
# ${SPARK_HOME}/bin/pyspark --packages com.databricks:spark-csv_2.11:1.4.0
# # ${SPARK_HOME}/bin/pyspark
# ${SPARK_HOME}/bin/spark-submit --master "local[*]" --driver-memory 5G Desktop/entity_resolution.py
# for running CSV from shell
# ${SPARK_HOME}/bin/spark-submit --executor-memory=12g --num-executors 10 --executor-cores 6 --master "local[*]" --packages com.databricks:spark-csv_2.11:1.4.0 ETL.py

