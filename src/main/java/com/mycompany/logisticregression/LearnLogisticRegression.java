/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.logisticregression;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import static org.apache.spark.sql.functions.col;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;

/**
 *
 * @author Doctorini
 */
public class LearnLogisticRegression {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {

        //Crear contexto Spark con nombre de app y la url del master, local[*]
        SparkConf conf = new SparkConf().setAppName("Base de datos cesarea").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        
        sc.setLogLevel("ERROR");
        
        //Para trabajar con Dataframes o Dataset(bd distribuidas)
        SparkSession spark = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();

        //crear dataset a partir de fichero csv
        Dataset<Row> df = spark.read().option("header", true).option("inferSchema", "true").csv("/media/maite/Data/Universidad/Cursos recibidos/2021-05 Intro a Big Data con Apache Spark/Tema 4/LogisticRegression/src/main/java/com/mycompany/logisticregression/caesarian.csv");

        //para ver esquema y ver los primeros 10 registros
        df.printSchema();

        Dataset<Row> logregdataall = df.select(col("Age"), col("Delivery_number"), col("Delivery_time"), col("Blood_Pressure"), col("Heart_Problem"), col("Cesarian").as("label"));

        logregdataall.show(10);

        //Eliminar valores ausentes
        Dataset<Row> logredata = logregdataall.na().drop();

        logredata.show(10);
        
        //Preparamos las siguientes transformaciones, para datos nominales
        StringIndexer delivery_numberIndexer = new StringIndexer().setInputCol("Delivery_number").setOutputCol("Delivery_numberIndex");
        StringIndexer delivery_timeIndexer = new StringIndexer().setInputCol("Delivery_time").setOutputCol("Delivery_timeIndex");
        StringIndexer blood_PressureIndexer = new StringIndexer().setInputCol("Blood_Pressure").setOutputCol("Blood_PressureIndex");
        StringIndexer heart_ProblemIndexer = new StringIndexer().setInputCol("Heart_Problem").setOutputCol("Heart_ProblemIndex");
        
        //delivery_numberIndexer.fit(logredata).transform(logredata).show();
        
        OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
                .setInputCols(new String[]{"Delivery_numberIndex", "Delivery_timeIndex",
            "Blood_PressureIndex", "Heart_ProblemIndex"})
                .setOutputCols(new String[]{"Delivery_numberVec", "Delivery_timeVec",
            "Blood_PressureVec", "Heart_ProblemVec"});
        
        //Creamos nuestro vector assembler con las columnas deseadas y la clase predictora
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"Age","Delivery_number","Delivery_time", "Blood_Pressure", "Heart_Problem"})
                .setOutputCol("features");

        //Dividimos los datos en dos partes 70 % para entrenar y 30 % para pruebas
        Dataset<Row>[] split = logredata.randomSplit(new double[]{0.7, 0.3}, 12345);
        
        split[0].show(10);

        //Creamos nuestro modelo de ML LogisticRegression
        
        LogisticRegression lr = new LogisticRegression();

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{delivery_numberIndexer, delivery_timeIndexer, blood_PressureIndexer, heart_ProblemIndexer, encoder,assembler,lr});
        
        //Búsqueda de hiperparametros
        ParamGridBuilder paramGrid = new ParamGridBuilder();
        paramGrid.addGrid(lr.regParam(), new double[]{0.1, 0.01,0.001,0.0001});

        //Buscamos hiper-parámetros, en este caso buscamos el parámetro regularizador.
        TrainValidationSplit trainValidationSplitLR = new TrainValidationSplit()
                .setEstimator(pipeline)
                .setEvaluator(new RegressionEvaluator())
                .setEstimatorParamMaps(paramGrid.build())
                .setTrainRatio(0.8);
        
        //Ejecutamos el entrenamiento
        TrainValidationSplitModel model = trainValidationSplitLR.fit(split[0]);

        //Ejecutamos las pruebas y lo guardamos en un dataset
        Dataset<Row> testResult = model.transform(split[1]);
        
        //Evaluamos metricas de rendimiento a partir de las pruebas
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(testResult);
        System.out.println("Test Error = " + (1.0 - accuracy));
        
        
        
        
        /**
         * Multilayer Perceptron
         */

        //Definimos la arquitectura con 5 neuronas en la capa de entrada (5 atributos)
        //4 y 3 como neuronas de las capa ocultas y 2 en la salida ya que son dos clasificaciones (efectuar cesarea o no)
        
        
        int[] layers = new int[]{5, 4, 3, 2};

        MultilayerPerceptronClassifier redNeuronal = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(1234L)
                .setMaxIter(100);
        redNeuronal.setFeaturesCol("featuresNormalized");
        redNeuronal.setLabelCol("cesarianIndex");

        //Discretizar la salida
        StringIndexer cesarianIndexer = new StringIndexer().setInputCol("label").setOutputCol("cesarianIndex");

        VectorAssembler assembler2 = new VectorAssembler()
                .setInputCols(new String[]{"Age", "Delivery_number", "Delivery_time", "Blood_Pressure", "Heart_Problem"})
                .setOutputCol("features");

        Normalizer normalizer = new Normalizer()
                .setInputCol("features")
                .setOutputCol("featuresNormalized")
                .setP(1.0);

        Dataset<Row>[] split2 = logregdataall.randomSplit(new double[]{0.7, 0.3}, 12345);
        System.out.println("schema\n\n" + split2[0].schema());
        System.out.println("schema\n\n" + split2[0].schema().json());

        Pipeline pipelineMLP = new Pipeline().setStages(new PipelineStage[]{cesarianIndexer, assembler2, normalizer, redNeuronal});

        //Configuramos el grid para buscar hiper-parámetros, en este caso de ejemplo máximo número de iteraciones
        ParamGridBuilder paramGridMLP = new ParamGridBuilder();
        paramGridMLP.addGrid(redNeuronal.stepSize(), new double[]{0.01, 0.001,0.0015});

        //Buscamos hiper-parámetros y ejecutamos el pipeline
        TrainValidationSplit trainValidationSplitMLP = new TrainValidationSplit()
                .setEstimator(pipelineMLP)
                .setEstimatorParamMaps(paramGridMLP.build())
                //Para el evaluador podemos elegir: BinaryClassificationEvaluator, ClusteringEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
                .setEvaluator(new BinaryClassificationEvaluator());

        TrainValidationSplitModel modelMLP = trainValidationSplitMLP.fit(split2[0]);
        Dataset<Row> resultMLP = modelMLP.transform(split2[1]);

        resultMLP.show();
        //Analizar métricas de rendimiento Accuracy y Confusion matrix
        MulticlassMetrics metrics3 = new MulticlassMetrics(resultMLP.select("prediction", "cesarianIndex"));

        System.out.println("Test set accuracy = " + metrics3.accuracy());
        System.out.println("Confusion matrix = \n" + metrics3.confusionMatrix());
        
    }

}
