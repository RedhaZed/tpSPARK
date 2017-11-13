package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer, StopWordsRemover,
StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator



object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/
     // PLEASE CHANGE THE PATH ACCORDING TO YOUR CONF
     //For a spark application it is possible to put the data in the "ressource" folder
   val df: DataFrame = spark
     .read
     .load("/home/redtpt/Téléchargements/funding-successful-projects-on-kickstarter/data/prepared_trainingset")

    df.show()

    /** TF-IDF **/
      //Tokenizer transforming text in tokens
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // removing unrelevant words 'the, a  , of ..."
    val stopwords = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    //Vectors of tokens counts
    val countvec = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("tokencount")

    //Getting the tfidf ratio
    val idf = new IDF()
      .setInputCol("tokencount")
      .setOutputCol("tfidf")

    // Indexing country and currency
    val country_ix = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country2_indexed")
      .setHandleInvalid("skip")

    val currency_ix = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency2_indexed")
      .setHandleInvalid("skip")

    // Creating the features vector
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country2_indexed", "currency2_indexed"))
      .setOutputCol("features")

    //Setting the LogisticRegression algorithm model
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    //Setting up the pipeline with all stages to run
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopwords, countvec, idf,country_ix,currency_ix,assembler,lr))


    // Spliting the data for training and test datasets
    val Array(trainingData, testData) = df.randomSplit(Array(0.9, 0.1))
    // Caching the dataframe to proccess up. Spark will not launch all the DAG from the beginning.
    trainingData.cache()
    testData.cache()

    // Tuning Hyper-parameters
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4,10e-2))
      .addGrid(countvec.minDF, Array(55.0,75.0, 95.0))
      .build()


    // We set the MulticlassClassificationEvaluator to evaluate each point of the grid. Using the f1 score.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    // We now treat the Pipeline as an Estimator, wrapping it in a trainValidationSplit instance.
    // Note : we could have use a CrossValidator to get better results. Since each point of the grid
    // is evaluted mutliple times (depending of the number of folds)
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)


      // Run cross-validation, and choose the best set of parameters.
    val model = trainValidationSplit.fit(trainingData)


    // Making predictions on the test set.
    val df_WithPredictions = model.transform(testData)

    //display the dataframe of predicted values
    df_WithPredictions.show()

    //Keeping only predictions and labels and evaluating the F1 score metric.
    val labels_predictions = df_WithPredictions.select("predictions", "final_status")
    val testF1Score = evaluator.evaluate(labels_predictions)
    println("F1 Score : "+testF1Score)

    //Count of different class by labels and predictions (similar to confusion matrix)
    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    //saving the final result dataframe and the model
    //PLEASE CHANGE THE PATH ACCORDING TO YOUR CONF
    df_WithPredictions.write.mode(SaveMode.Overwrite)
      .parquet("/home/redtpt/Téléchargements/funding-successful-projects-on-kickstarter/data/prediction4")
    model.write.overwrite().save("/home/redtpt/Téléchargements/funding-successful-projects-on-kickstarter/models")
    //displaying the df
    labels_predictions.head()
      }


}
