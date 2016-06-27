from pyspark import SparkConf, SparkContext,SQLContext, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
import sys, operator
import json
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
#inputs = sys.argv[1]
#output = sys.argv[2]
inputs = "/Users/saeeds/Downloads/housingprices/Datasets/"
output = "/users/saeeds/Downloads/housingprices/Datasets/ETLData"


conf = SparkConf().setAppName('housingprice')
sc = SparkContext(conf=conf)

sqlContext = SQLContext(sc)
# Check all columns of a given dataframe to have a valid value 
# it returns -1 if there is no data, otherwise it returns 1     
def checkDF(df):
    l = df.columns
    for col in l : 
        if (df.filter(df[col] =="").count() > 0 ): 
            return -1
    return 1    
# ETL postalCode column ===> FSA
# only first part of postal code is considered in our model and they change to numeric value using hot encoding    
def getFSA(postalcode):
    postalcode = postalcode.strip() 
    j = postalcode.index(' ') 
    return postalcode[0:j]


def makeCode(i, len):
    code = '0' * len
    code = code[0:i] + '1' + code[i+1:]
    return code


# ETL zoneName, zoneCat
# set null values in Zone category with proper value from Zone Name
# all characters before '-' in zone name is the code for zone category 
def getZoneCat(zoneName):
    if (zoneName.find('-') == -1) :
        return 'CD'  
    j  = zoneName.index('-')  
    return zoneName[0:j] 
     
def fixdate(convVal):
        a = convVal.split(" ")
        dates = a[0].split("/")
        alldate = "20"+dates[2]+'/'+dates[0]
        return (alldate,a[1])
        
def filterYear(dates):
        a = dates.split('/')
        if (a[1]=='2016'):
            return False
        else:
            return True
            
def processDate(df):
        def splitMonth(cols):
         a = cols.split('/')
         return a[1]
        def splitYear(cols):
         a = cols.split('/')
         return a[0]
         
        fUDF = udf(splitMonth, StringType())
        df1 =  df.withColumn("month", fUDF('date'))
        fUDFyear = udf(splitYear, StringType())
        return df1.withColumn("year", fUDFyear('date'))
        
def splitYear(cols):
         a = cols.split('/')
         return a[0]
        
taxReportSchema = StructType([
        StructField('PID', StringType(), False),
        StructField('legalType', StringType(), False),
        StructField('folio', StringType(), False),
        StructField('coordinates', StringType(), True),
        StructField('zoneName', StringType(), False),
        StructField('zoneCat', StringType(), True),
        StructField('lot', StringType(), True),
        StructField('block', StringType(), True),
        StructField('plan', StringType(), True),
        StructField('districtLot', StringType(), True),
        StructField('fCivicNum', StringType(), True),
        StructField('tCivicNum', StringType(), True),
        StructField('streetName', StringType(), True),
        StructField('postalCode', StringType(), True),
        StructField('NLegalName1', StringType(), True),
        StructField('NLegalName2', StringType(), True),
        StructField('NLegalName3', StringType(), True),
        StructField('NLegalName4', StringType(), True),
        StructField('NLegalName5', StringType(), True),
        StructField('curVal', StringType(), True),
        StructField('curImpVal', StringType(), True),
        StructField('taxAssess', StringType(), True),
        StructField('preVal', StringType(), True),
        StructField('preImpVal', StringType(), True),
        StructField('yearBuilt', StringType(), True),
        StructField('bigImpYear', StringType(), True),
        StructField('taxLevy', StringType(), True),
        StructField('neighbourhoodCode', StringType(), True),
    ])
conversionSchema = StructType([
        StructField('date', StringType(), False),
        StructField('USD', StringType(), False),
        StructField('rate', StringType(), False),
        StructField('reciprate', StringType(), False),
    ])
crudeoilSchema = StructType([
        StructField('date', DateType(), False),
        StructField('oilprice', StringType(), False),
    ])    

#Reading the Tax Report Dataset; total records = 200728
filename = inputs + "taxreport/property_tax_report_csv2016.csv"
df = sqlContext.read.format('com.databricks.spark.csv').options(header='true').schema(taxReportSchema).load(filename)
df.registerTempTable("taxreport2016")
# Select records that have valid PID and Valid postalCode; total record = 198040
df2016 = sqlContext.sql("SELECT PID, legalType, zoneName, zoneCat, postalCode, curVal, curImpVal, taxAssess, preVal, preImpVal FROM taxreport2016 WHERE (not (PID ='') AND Length(LTRIM(RTRIM(postalCode)))=7 ) ").dropDuplicates(['PID']).cache()
df2016.registerTempTable("tax2016")
print df2016.count() #198011
extractFSA = udf(getFSA, StringType())
df2016 = df2016.withColumn('FSA', extractFSA(df2016.postalCode)).cache()
df2016.registerTempTable("postal") 
FSAlist = sqlContext.sql("SELECT distinct FSA FROM  postal").map(lambda x: str(x[0])).collect() 

values = []
encoding = []
n = len(FSAlist)
for i in range(n) :
     values.append(FSAlist[i])
     encoding.append(makeCode(i, n))
df2016 = df2016.replace(values, encoding ,'FSA').cache()
replaceValue = udf(getZoneCat, StringType())
df2016 = df2016.withColumn('zoneCat', F.when(df2016.zoneCat=='', replaceValue(df2016.zoneName)).otherwise(df2016.zoneCat)).cache()

# zoneName is not null or empty value
zname = ['One Family Dwelling', 'Two Family Dwelling', 'Multiple Family Dwelling', 'Limited Agricultural', 'Commercial', 'Comprehensive Development', 'Light Industrial', 'Industrial', 'Historic Area' ]
zcode = ['RS', 'RT', 'RM', 'RA', 'C', 'CD','IC', 'M', 'HA']
zhotEncode = ['100000000', '010000000', '001000000', '000100000', '000010000', '000001000', '000000100', '000000010', '000000001']
df2016 = df2016.replace(zname, zcode,'zoneCat')

# changing the categorical data to numerical data using hot encoding 
df2016 = df2016.replace(zcode, zhotEncode,'zoneCat')
                       
# ETL legalType column:
# changing the categorical data to numerical data using hot encoding; all records has regalType.                         
df2016 = df2016.replace(['LAND', 'STRATA', 'OTHER'], ['100','010','001'],'legalType')  

df2016.registerTempTable("data2016") 
df2016 = sqlContext.sql("SELECT PID, legalType, zoneCat, curVal, curImpVal, taxAssess as year, preVal, preImpVal, FSA FROM  data2016").cache()
training, test = df2016.randomSplit([0.6, 0.4], seed=0)
print training.take(10)
if (checkDF(df2016) == -1):
	print "ERROR: EMPTY VALUE FOR DATA IN DATAFRAME"           
#allDF = df2016

for i in [2014, 2015]: 
    filename = inputs + "taxreport/property_tax_report_csv" + str(i) +".csv"
    df = sqlContext.read.format('com.databricks.spark.csv').options(header='true').schema(taxReportSchema).load(filename)
    
    df.registerTempTable("taxreport"+str(i))
    
    validDF = sqlContext.sql("SELECT PID, legalType, zoneName, zoneCat, postalCode, curVal, curImpVal, taxAssess, preVal, preImpVal FROM taxreport" + str(i) + " WHERE (not (PID ='') AND Length(LTRIM(RTRIM(postalCode)))=7  AND not (zoneName=''))").dropDuplicates(['PID']).cache()
    validDF.registerTempTable("tax"+str(i))
    # ETL postalCode column ===> FSA
    # only first part of postal code is considered in our model and they change to numeric value using hot encoding 
     
    validDF = validDF.withColumn('FSA', extractFSA(validDF.postalCode))
    validDF = validDF.replace(values, encoding ,'FSA').cache() 
    
    # ETL zoneName, zoneCat                 
    # set null values in Zone category with proper value from Zone Name
    # all characters before '-' in zone name is the code for zone category 
    validDF = validDF.withColumn('zoneCat', F.when(validDF.zoneCat=='', replaceValue(validDF.zoneName)).otherwise(validDF.zoneCat)).cache()
    
    # zoneName is not null or empty value
    validDF = validDF.replace(zname , zcode,'zoneCat')

    # changing the categorical data to numerical data using hot encoding 
    validDF = validDF.replace(zcode, zhotEncode,'zoneCat')
                       
    # ETL legalType column:
    # changing the categorical data to numerical data using hot encoding; all records has regalType.                         
    validDF = validDF.replace(['LAND', 'STRATA', 'OTHER'], ['100','010','001'],'legalType')  
    
    validDF.registerTempTable("data"+str(i)) 
    validDF = sqlContext.sql("SELECT PID, legalType, zoneCat, curVal, curImpVal, taxAssess as year, preVal, preImpVal, FSA FROM  data"+str(i)) 
    
    if (checkDF(validDF) == -1) :
		print "ERROR: EMPTY VALUE FOR DATA IN DATAFRAME"
         
    # union all with the rest of data
    allDF = df2016.unionAll(validDF)


# extracting historical data from 2006 to 2013
# Historical data don't have zone category and previous values
# In order to ETL, missing data are extracted from mainDF and injected     
taxReportSchema2 = StructType([
        StructField('PID', StringType(), False),
        StructField('legalType', StringType(), False),
        StructField('folio', StringType(), False),
        StructField('coordinates', StringType(), True),
        StructField('lot', StringType(), True),
        StructField('block', StringType(), True),
        StructField('plan', StringType(), True),
        StructField('districtLot', StringType(), True),
        StructField('fCivicNum', StringType(), True),
        StructField('tCivicNum', StringType(), True),
        StructField('streetName', StringType(), True),
        StructField('postalCode', StringType(), True),
        StructField('NLegalName1', StringType(), True),
        StructField('NLegalName2', StringType(), True),
        StructField('NLegalName3', StringType(), True),
        StructField('NLegalName4', StringType(), True),
        StructField('NLegalName5', StringType(), True),
        StructField('curVal', StringType(), True),
        StructField('curImpVal', StringType(), True),
        StructField('taxAssess', StringType(), True),
        StructField('yearBuilt', StringType(), True),
        StructField('bigImpYear', StringType(), True),
        StructField('taxLevy', StringType(), True),
        StructField('neighbourhoodCode', StringType(), True),
    ])  

# doing ETL for 2006
filename = inputs + "taxreport/property_tax_report_csv2006.csv"
df = sqlContext.read.format('com.databricks.spark.csv').options(header='true').schema(taxReportSchema2).load(filename)
df.registerTempTable("taxreport2006")
# Select records that have valid PID and Valid postalCode; total record = 
preDF = sqlContext.sql("SELECT PID, legalType, postalCode, curVal, curImpVal, taxAssess AS year FROM taxreport2006 WHERE (not (PID ='') AND Length(LTRIM(RTRIM(postalCode)))=7 ) ").dropDuplicates(['PID']).cache()
preDF.registerTempTable("tax2006")
print preDF.count() 
# ETL postalCode column ===> FSA
# only first part of postal code is considered in our model and they change to numeric value using hot encoding    
preDF = preDF.withColumn('FSA', extractFSA(preDF.postalCode))
preDF = preDF.replace(values, encoding ,'FSA').cache() 
# ETL legalType column:
# changing the categorical data to numerical data using hot encoding; all records has regalType.                         
preDF = preDF.replace(['LAND', 'STRATA', 'OTHER'], ['100','010','001'],'legalType')    # 191171

# set preVal and preImpVal from the preDF with 99 % of current value, because we have no data in this regards
preDF = preDF.withColumn('preVal', preDF.curVal * 0.99)
preDF = preDF.withColumn('preImpVal', preDF.curImpVal * 0.99).cache()

# Add zoneCat column into preDF                
# data comes from df2016 with the same PID
# jointDF = preDF.join(mainDF, preDF.PID==mainDF.PID).select(mainDF.PID, mainDF.zoneCat).distinct().cache()  # 191238
#ADD LATER FOR TESTING________________________________________>
preDF = preDF.join(allDF, preDF.PID==allDF.PID).select(preDF.PID, preDF.legalType, allDF.zoneCat, preDF.curVal, preDF.curImpVal, preDF.year, preDF.preVal, preDF.preImpVal, preDF.FSA).dropDuplicates(['PID']).cache() #190691

# check all columns to have a valid value
if (checkDF(preDF) == -1) :
	print "ERROR: EMPTY VALUE FOR DATA IN DATAFRAME"
     
allHDF = preDF.cache()
#outalldata = allHDF.repartition(40).rdd.map(lambda w: str(w.PID)+" "+str(w.legalType)+" "+str(w.FSA)+" "+ str(w.curVal)).coalesce(1)
#outalldata.saveAsTextFile(output) 
for i in range(2007,2014):
	filename = inputs + "taxreport/property_tax_report_csv"+ str(i)+".csv"
	df = sqlContext.read.format('com.databricks.spark.csv').options(header='true').schema(taxReportSchema2).load(filename)
	df.registerTempTable("taxreport"+ str(i))
	curDF = sqlContext.sql("SELECT PID, legalType, postalCode, curVal, curImpVal, taxAssess AS year FROM taxreport" + str(i) + " WHERE (not (PID ='') AND Length(LTRIM(RTRIM(postalCode)))=7 ) ")
	curDF = curDF.dropDuplicates(['PID']).cache()
	curDF.registerTempTable("tax"+str(i))
	# print curDF.count() 
	# ETL postalCode column ===> FSA
	# only first part of postal code is considered in our model and they change to numeric value using hot encoding    
	curDF = curDF.withColumn('FSA', extractFSA(curDF.postalCode))
	curDF = curDF.replace(values, encoding ,'FSA').cache() 
	# ETL legalType column:
	# changing the categorical data to numerical data using hot encoding; all records has regalType.                         
	curDF = curDF.replace(['LAND', 'STRATA', 'OTHER'], ['100','010','001'],'legalType')    # 191171
	
    # set preVal and preImpVal from the preDF
	curDF = curDF.join(preDF, curDF.PID==preDF.PID).select(curDF.PID, curDF.legalType, curDF.curVal, curDF.curImpVal, curDF.year, (preDF.curVal).alias('preVal'), (preDF.curImpVal).alias('preImpVal'), curDF.FSA).dropDuplicates(['PID']) 
    
    # Add zoneCat column into curDF                
	# data comes from df2016 with the same PID
	# jointDF = curDF.join(mainDF, curDF.PID==mainDF.PID).select(mainDF.PID, mainDF.zoneCat).distinct().cache()  
	# count 191238
	#ADDD LATER FOR TESTINGGGG_______________________________________>
	curDF = curDF.join(allDF, curDF.PID==allDF.PID).select(curDF.PID, curDF.legalType, allDF.zoneCat, curDF.curVal, curDF.curImpVal, curDF.year, curDF.preVal, curDF.preImpVal, curDF.FSA).dropDuplicates(['PID']).cache()
	if ( checkDF(curDF) == -1 ):
		print "ERROR: EMPTY VALUE FOR DATA IN DATAFRAME"
    # keep the curDF as preDF
	#preDF = curDF.cache() 
	allHDF = allHDF.unionAll(curDF)
print allHDF.count()
mainDF = allHDF.unionAll(allDF).cache()       
#outalldata = mainDF.repartition(40).rdd.map(lambda w: str(w.PID)+" "+str(w.legalType)+" "+str(w.FSA)+" "+ str(w.curVal)+" "+ str(w.year)).coalesce(1)
#outalldata.saveAsTextFile(output)
#mainDF.write.parquet('/users/saeeds/Downloads/housingprices/Datasets/ETLDataParquet')    


#Reading the CAN to USD conversion dataset
conversion = sqlContext.read.format('com.databricks.spark.csv').options(header='true').schema(conversionSchema).load(inputs+"conversion")
conversion.registerTempTable("Conversion")
#Selecting only the date and rate
conversionrate = sqlContext.sql("SELECT date,rate FROM Conversion WHERE rate regexp '^[0-9]+'")
conversionRDD = conversionrate.repartition(40).rdd.map(lambda w: (w.date+" "+w.rate))
conversiondates = conversionRDD.map(fixdate).map(lambda l: Row(date=l[0], rate=l[1]))
schemaConv = sqlContext.inferSchema(conversiondates)
schemaConv.registerTempTable("ConversionDate")
ConverDF = sqlContext.sql(" SELECT date,CAST(AVG(rate) AS DECIMAL(4,2)) as conversionrate FROM ConversionDate WHERE rate IS NOT NULL GROUP BY date")
ConverDate = processDate(ConverDF)
ConverDate.registerTempTable("ConverDates")
#Selecting the date on M/Y format and oilprice
ConverDF = sqlContext.sql("SELECT year,CAST(AVG(conversionrate) AS DECIMAL(4,2)) as conversionrate FROM ConverDates GROUP BY year")

#Reading the Canada Crude oil price dataset
crudeoil = sc.textFile(inputs+"crudeoil")
crudeoilRDD = crudeoil.map(lambda l: l.split()).map(lambda l: Row(date=l[0], oilprice=l[1]))
crudeoilDF = sqlContext.inferSchema(crudeoilRDD)
crudeoilDF.registerTempTable("crudeoil")
#Selecting the date on M/Y format and oilprice
oilpriceall = sqlContext.sql("SELECT DATE_FORMAT(date,'Y') as date,oilprice FROM crudeoil WHERE oilprice!='.' ")
oilpriceall.registerTempTable('oilpriceall')
oilprice = sqlContext.sql("SELECT date,AVG(oilprice) as oilprice FROM oilpriceall GROUP BY date ")

#Reading the interestrate of BC Dataset
interestRate = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load(inputs+"interestrate")
interestRate.registerTempTable("interest")
#Selecting the date and 5-year fixed mortgage price from the dataset
interestDF = sqlContext.sql("SELECT DATE_FORMAT(date,'Y') as date,CAST(`5y-fixed-posted` AS DECIMAL(4,2)) AS interestrate FROM interest WHERE date >='2006-01'")
interestDF.registerTempTable("allrates")

#Getting the average of each month on days whose value is not null.
avgInterest = sqlContext.sql(" SELECT date,CAST(AVG(interestrate) AS DECIMAL(4,2)) as interestrates FROM allrates WHERE interestrate IS NOT NULL GROUP BY date")


joinedTable = avgInterest.join(oilprice,(avgInterest['date']==oilprice['date'])).select(avgInterest['date'],avgInterest['interestrates'],oilprice['oilprice'])

JoinedConversion = joinedTable.join(ConverDF,(joinedTable['date']==ConverDF['year'])).select(joinedTable['date'].alias('year'),joinedTable['interestrates'],joinedTable['oilprice'],ConverDF['conversionrate'])
JoinedConversion.registerTempTable("joinedConversion")
JoinedConversion.registerTempTable("allmonths")

modelDF = JoinedConversion.join(mainDF, JoinedConversion.year==mainDF.year).select(mainDF.legalType, mainDF.zoneCat, mainDF.curVal, mainDF.curImpVal, mainDF.year, mainDF.preVal, mainDF.preImpVal, mainDF.FSA,JoinedConversion.interestrates,JoinedConversion.oilprice,JoinedConversion.conversionrate)
training, test = modelDF.randomSplit([0.6, 0.4], seed=0)
#outalldata = mainDF.repartition(40).rdd.map(lambda w: str(w.PID)+" "+str(w.legalType)+" "+str(w.FSA)+" "+ str(w.curVal)+" "+ str(w.year)).coalesce(1)
mainDF.show()
training.write.parquet(output+"/training")
test.write.parquet(output+"/testing")
# setenv SPARK_HOME /Volumes/bshadgar/Spark/spark-1.5.2-bin-hadoop2.6
## setenv SPARK_HOME /Volumes/saeeds/spark-1.5.1-bin-hadoop2.6
# # ${SPARK_HOME}/bin/pyspark
# ${SPARK_HOME}/bin/spark-submit --master "local[*]" --driver-memory 5G Desktop/entity_resolution.py
# for running CSV from shell
# ${SPARK_HOME}/bin/spark-submit --executor-memory=12g --num-executors 10 --executor-cores 6 --master "local[*]" --packages com.databricks:spark-csv_2.11:1.4.0 ETL.py