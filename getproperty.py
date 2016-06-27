from pyspark import SparkConf, SparkContext,SQLContext, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
from pyspark.sql.functions import *
import sys, operator
import json


#inputs = sys.argv[1]
#output = sys.argv[2]

inputs = "/Users/saeeds/Downloads/Datasets/"
#output = "/users/bshadgar/Desktop/out"


def main():
    conf = SparkConf().setAppName('housingprice')
    sc = SparkContext(conf=conf)

    sqlContext = SQLContext(sc)
    taxreportSchema = StructType([
        StructField('PID', StringType(), False),
        StructField('Legal_Type', StringType(), False),
        StructField('FOLIO', StringType(), False),
        StructField('Coordinates', StringType(), True),
        StructField('ZoneName', StringType(), True),
        StructField('ZoneCat', StringType(), True),
        StructField('LOT', StringType(), True),
        StructField('Block', StringType(), True),
        StructField('plan', StringType(), True),
        StructField('DisLot', StringType(), True),
        StructField('FCiviNum', StringType(), True),
        StructField('TCiviNum', StringType(), True),
        StructField('StreetName', StringType(), True),
        StructField('PostalCode', StringType(), True),
        StructField('NLegalName1', StringType(), True),
        StructField('NLegalName2', StringType(), True),
        StructField('NLegalName3', StringType(), True),
        StructField('NLegalName4', StringType(), True),
        StructField('NLegalName5', StringType(), True),
        StructField('CurVal', StringType(), True),
        StructField('CurImpVal', StringType(), True),
        StructField('Taxassess', StringType(), True),
        StructField('prevVal', StringType(), True),
        StructField('prevImpVal', StringType(), True),
        StructField('YearBuilt', StringType(), True),
        StructField('BigImpYear', StringType(), True),
        StructField('Tax_levy', StringType(), True),
        StructField('NeighbourhoodCode', StringType(), True),
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
        df1 =  df.withColumn("month", fUDF('year'))
        fUDFyear = udf(splitYear, StringType())
        return df1.withColumn("year", fUDFyear('year'))
    #Reading the Tax Report Dataset
    taxreportinfo = sqlContext.read.format('com.databricks.spark.csv').options(header='true').schema(taxreportSchema).load(inputs+"taxreport/test")
    taxreportinfo.registerTempTable("taxreport")
    #Selecting the price,TaxAssessment Year and Postalcode of each property
    propertyVal = sqlContext.sql("SELECT CurVal, Taxassess, PostalCode FROM taxreport")
    propertyVal.registerTempTable("propertyVal")
    #Reading the CAN to USD conversion dataset
    conversion = sqlContext.read.format('com.databricks.spark.csv').options(header='true').schema(conversionSchema).load(inputs+"conversion")
    conversion.registerTempTable("Conversion")
    #Selecting only the date and rate
    conversionrate = sqlContext.sql("SELECT date,rate FROM Conversion WHERE rate regexp '^[0-9]+'")
    conversionRDD = conversionrate.repartition(40).rdd.map(lambda w: (w.date+" "+w.rate))
    conversiondates = conversionRDD.map(fixdate).filter(lambda (w,x):filterYear(w)).map(lambda l: Row(date=l[0], rate=l[1]))
    schemaConv = sqlContext.inferSchema(conversiondates)
    schemaConv.registerTempTable("ConversionDate")
    ConverDF = sqlContext.sql(" SELECT date,CAST(AVG(rate) AS DECIMAL(4,2)) as conversionrate FROM ConversionDate WHERE rate IS NOT NULL GROUP BY date")
    ConverDF.cache()
    #Reading the Canada Crude oil price dataset
    crudeoil = sc.textFile(inputs+"crudeoil")
    crudeoilRDD = crudeoil.map(lambda l: l.split()).map(lambda l: Row(date=l[0], oilprice=l[1]))
    crudeoilDF = sqlContext.inferSchema(crudeoilRDD)
    crudeoilDF.registerTempTable("crudeoil")
    #Selecting the date on M/Y format and oilprice
    oilprice = sqlContext.sql("SELECT DATE_FORMAT(date,'Y/M') as date,oilprice FROM crudeoil")
    oilprice.registerTempTable('oilprice')
    #Reading the interestrate of BC Dataset
    interestRate = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load(inputs+"interestrate")
    interestRate.registerTempTable("interest")
    #Selecting the date and 5-year fixed mortgage price from the dataset
    interestDF = sqlContext.sql("SELECT DATE_FORMAT(date,'Y/M') as date,CAST(`5y-fixed-posted` AS DECIMAL(4,2)) AS interestrate FROM interest WHERE date >='2006-01' AND date <= '2015-12'")
    interestDF.registerTempTable("allrates")
    #Getting the average of each month on days whose value is not null.
    avgInterest = sqlContext.sql(" SELECT date,AVG(interestrate) as interestrates FROM allrates WHERE interestrate IS NOT NULL GROUP BY date")
    avgInterest.cache()
    joinedTable = avgInterest.join(oilprice,(avgInterest['date']==oilprice['date'])).select(avgInterest['date'],avgInterest['interestrates'],oilprice['oilprice'])
    JoinedConversion = joinedTable.join(ConverDF,(joinedTable['date']==ConverDF['date'])).select(joinedTable['date'].alias('year'),joinedTable['interestrates'],joinedTable['oilprice'],ConverDF['conversionrate'])
    JoinedConversion.registerTempTable("joinedConversion")
    ls = processDate(JoinedConversion)
    ls.show()
    #splitYear = sqlContext.sql('SELECT SUBSTRING_INDEX(SUBSTRING_INDEX(date, ' ', 1), ' ', -1) as memberfirst FROM joinedConversion ')
    #splitYear.show()
    #interestRDD = FeatureModel.repartition(40).rdd.map(lambda w: (w.date+" "+str(w.interestrates)+' '+str(w.conversionrate))).coalesce(1)
    #interestRDD.saveAsTextFile('/Users/saeeds/Downloads/output')
    #print oilprice.count()
if __name__ == "__main__":
    main()
#${SPARK_HOME}/bin/spark-submit --master "local[*]" --packages com.databricks:spark-csv_2.11:1.4.0 getproperty.py
#setenv SPARK_HOME /Volumes/personal/saeeds-irmacs/spark-1.5.1-bin-hadoop2.6