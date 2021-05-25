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
 * @author Maite Sánchez Fornaris & Jesus Machado Oramas
 */
public class CylinderBandsAnalysis {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("Cylinder Bands Dataset").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        
        sc.setLogLevel("ERROR");
        
        SparkSession spark = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();
        
        // 1. Utilizando el fichero en formato csv provisto por su profesor realice las siguientes acciones:
        // a. Cree un Dataframe a partir del fichero e imprima en pantalla su esquema y las 10 primeras filas.
        // b. Imprima en pantalla los posibles valores que toma el atributo predictor o etiqueta de clasificación.
        // c. Realice las transformaciones sobre los datos para eliminar valores ausentes, datos anómalos, etc.
        // d. Aplique las transformaciones necesarias sobre los datos que contengan valores nominales, mediante técnicas de extracción de características.

        // Cargar el dataset, luego eliminar los valores nulos y convertir todas las variables nominales a numerica mediante codificacion
        Dataset<Row> df = spark.read().option("header", false).option("inferSchema", "true").csv("/home/maite/Documents/LogisticRegression/src/main/java/com/mycompany/logisticregression/bands.data");
        
        df.printSchema();  // inciso a
        df.show(10);       // inciso a
        System.out.println("Cantidad de filas del dataframe: " + df.count());
        
        // Eliminar las filas con valores nulos en el dataframe
        
        
        df = df.na().drop(); // inciso c
        System.out.println("Cantidad de filas del dataframe despues de eliminar valores nulos: " + df.count());
        
        //Mostrar los posibles valores que puede tomar el campo objetivo (target) que vamos a predecir, en este caso es band_type
        df.groupBy("_c39").count().show();    // inciso b
        
        // Transformar los datos que tengas valores nominales, usar tecnicas de extraccion de caracteristicas
        
        
        
        StringIndexer grain_screenedIndexer = new StringIndexer().setInputCol("_c4").setOutputCol("grain_screenedIndex");                 //1
        StringIndexer ink_colorIndexer = new StringIndexer().setInputCol("_c5").setOutputCol("ink_colorIndex");                           //2
        StringIndexer proof_ctd_inkIndexer = new StringIndexer().setInputCol("_c6").setOutputCol("proof_ctd_inkIndex");                   //3
        StringIndexer blade_mfgIndexer = new StringIndexer().setInputCol("_c7").setOutputCol("blade_mfgIndex");                           //4
        StringIndexer cylinder_divisionIndexer = new StringIndexer().setInputCol("_c8").setOutputCol("cylinder_divisionIndex");           //5
        StringIndexer paper_typeIndexer = new StringIndexer().setInputCol("_c9").setOutputCol("paper_typeIndex");                        //6
        StringIndexer ink_typeIndexer = new StringIndexer().setInputCol("_c10").setOutputCol("ink_typeIndex");                              //7
        StringIndexer direct_steamIndexer = new StringIndexer().setInputCol("_c11").setOutputCol("direct_steamIndex");                  //8
        StringIndexer solvent_typeIndexer = new StringIndexer().setInputCol("_c12").setOutputCol("solvent_typeIndex");                  //9
        StringIndexer type_on_cylinderIndexer = new StringIndexer().setInputCol("_c13").setOutputCol("type_on_cylinderIndex");      //10
        StringIndexer press_typeIndexer = new StringIndexer().setInputCol("_c14").setOutputCol("press_typeIndex");                        //11
        StringIndexer pressIndexer = new StringIndexer().setInputCol("_c15").setOutputCol("pressIndex");                                       //12
        StringIndexer unit_numberIndexer = new StringIndexer().setInputCol("_c16").setOutputCol("unit_numberIndex");                     //13
        StringIndexer cylinder_sizeIndexer = new StringIndexer().setInputCol("_c17").setOutputCol("cylinder_sizeIndex");               //14
        StringIndexer paper_mill_locationIndexer = new StringIndexer().setInputCol("_c18").setOutputCol("paper_mill_locationIndex"); //15
        StringIndexer plating_tankIndexer = new StringIndexer().setInputCol("_c19").setOutputCol("plating_tankIndex");                  //16
        StringIndexer proof_cutIndexer = new StringIndexer().setInputCol("_c20").setOutputCol("proof_cutIndex");                           //17
        StringIndexer viscosityIndexer = new StringIndexer().setInputCol("_c21").setOutputCol("viscosityIndex");                           //18
        StringIndexer caliperIndexer = new StringIndexer().setInputCol("_c22").setOutputCol("caliperIndex");                                 //19
        StringIndexer ink_temperatureIndexer = new StringIndexer().setInputCol("_c23").setOutputCol("ink_temperatureIndex");         //20
        StringIndexer humifityIndexer = new StringIndexer().setInputCol("_c24").setOutputCol("humifityIndex");                              //21
        StringIndexer roughnessIndexer = new StringIndexer().setInputCol("_c25").setOutputCol("roughnessIndex");                           //22
        StringIndexer blade_pressureIndexer = new StringIndexer().setInputCol("_c26").setOutputCol("blade_pressureIndex");            //23
        StringIndexer varnish_pctIndexer = new StringIndexer().setInputCol("_c27").setOutputCol("varnish_pctIndex");                     //24
        StringIndexer press_speedIndexer = new StringIndexer().setInputCol("_c28").setOutputCol("press_speedIndex");                     //25
        StringIndexer ink_pctIndexer = new StringIndexer().setInputCol("_c29").setOutputCol("ink_pctIndex");                                 //26
        StringIndexer solvent_pctIndexer = new StringIndexer().setInputCol("_c30").setOutputCol("solvent_pct");                          //27
        StringIndexer esa_voltageIndexer = new StringIndexer().setInputCol("_c31").setOutputCol("esa_voltageIndex");                     //28
        StringIndexer esa_amperageIndexer = new StringIndexer().setInputCol("_c32").setOutputCol("esa_amperageIndex");                  //29
        StringIndexer waxIndexer = new StringIndexer().setInputCol("_c33").setOutputCol("waxIndex");                                             //30
        StringIndexer hardenerIndexer = new StringIndexer().setInputCol("_c34").setOutputCol("hardenerIndex");                              //31
        StringIndexer roller_durometerIndexer = new StringIndexer().setInputCol("_c35").setOutputCol("roller_durometerIndex");      //32
        StringIndexer current_densityIndexer = new StringIndexer().setInputCol("_c36").setOutputCol("current_densityIndex");         //33
        StringIndexer anode_space_ratioIndexer = new StringIndexer().setInputCol("_c37").setOutputCol("anode_space_ratioIndex");   //34
        StringIndexer chrome_contentIndexer = new StringIndexer().setInputCol("_38").setOutputCol("chrome_contentIndex");            //35
        
        // Discretizar tambien la salida
        StringIndexer band_typeIndexer = new StringIndexer().setInputCol("_c39").setOutputCol("band_typeIndex");                           // label codification
        System.out.println(grain_screenedIndexer);
        OneHotEncoderEstimator encoder = new OneHotEncoderEstimator().setInputCols(new String[]{"grain_screenedIndex",
            "ink_colorIndex", "proof_ctd_inkIndex", "blade_mfgIndex", "cylinder_divisionIndex", "paper_typeIndex",
            "ink_typeIndex", "direct_steamIndex", "solvent_typeIndex", "type_on_cylinderIndex", "press_typeIndex",
            "pressIndex", "unit_numberIndex", "cylinder_sizeIndex", "paper_mill_locationIndex", "plating_tankIndex",
            "proof_cutIndex", "viscosityIndex", "caliperIndex", "ink_temperatureIndex", "humifityIndex", "roughnessIndex",
            "blade_pressureIndex", "varnish_pctIndex", "press_speedIndex", "ink_pctIndex", "solvent_pctIndex", "esa_voltageIndex",
            "esa_amperageIndex", "waxIndex", "hardenerIndex", "roller_durometerIndex", "current_densityIndex", "anode_space_ratioIndex",
            "chrome_contentIndex"})
                .setOutputCols(new String[]{"grain_screenedVec", "ink_colorVec", "proof_ctd_inkVec", "blade_mfgVec", "cylinder_divisionVec",
                    "paper_typeVec", "ink_typeVec", "direct_steamVec", "solvent_typeVec", "type_on_cylinderVec", "press_typeVec", "pressVec",
                    "unit_numberVec", "cylinder_sizeVec", "paper_mill_locationVec", "plating_tankVec", "proof_cutVec", "viscosityVec",
                    "caliperVec", "ink_temperatureVec", "humifityVec", "roughnessVec", "blade_pressureVec", "varnish_pctVec",
                    "press_speedVec", "ink_pctVec", "solvent_pctVec", "esa_voltageVec", "esa_amperageVec", "waxVec", "hardenerVec",
                    "roller_durometerVec", "current_densityVec", "anode_space_ratioVec", "chrome_contentVec"});
       
        //Hacer 
        // 2. Seleccione al menos tres algoritmos de aprendizaje automático de acuerdo al problema identificado en el dataset y realice las siguientes acciones:
        // 1er modelo: Multilayer perceptron
        // 2do modelo: Random forest classifier
        // 3er modelo: matrix factorization
        // a. Cree un VectorAssembler a partir de los datos pre-procesados y divida de forma aleatoria el conjunto en dos partes un 70 % para entrenamiento y el 30 % para pruebas.
        //VectorAssembler assembler = new VectorAssembler().setInputCols(new String[]{"grain_screened", "ink_color", "proof_ctd_ink", "blade_mfg",
        //    "cylinder_division", "paper_type", "ink_type", "direct_steam", "solvent_type", "type_on_cylinder", "press_type", "press", "unit_number",
        //    "cylinder_size", "paper_mill_location", "plating_tank", "proof_cut", "viscosity", "caliper", "ink_temperature", "humifity", "roughness",
        //    "blade_pressure", "varnish_pct", "press_speed", "ink_pct", "solvent_pct", "esa_voltage", "esa_amperage", "wax", "hardener", "roller_durometer",
        //    "current_density", "anode_space_ratio", "chrome_content"}).setOutputCol("features");
        
        //Dataset<Row>[] split = logregdataall.randomSplit(new double[]{0.8, 0.2}, 1);
        //split[0].show(10);
        
        
        // b. Entrene cada modelo elegido con dicha entrada y ajuste los hiper-parámetros correspondientes de forma automática.
        //LogisticRegression lr = new LogisticRegression();
        
        //Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{grain_screenedIndexer, ink_colorIndexer, proof_ctd_inkIndexer,
        //    blade_mfgIndexer, cylinder_divisionIndexer, paper_typeIndexer, ink_typeIndexer, direct_steamIndexer, solvent_typeIndexer,
        //    type_on_cylinderIndexer, press_typeIndexer, pressIndexer, unit_numberIndexer, cylinder_sizeIndexer, paper_mill_locationIndexer,
        //    plating_tankIndexer, proof_cutIndexer, viscosityIndexer, caliperIndexer, ink_temperatureIndexer, humifityIndexer, roughnessIndexer,
        //    blade_pressureIndexer, varnish_pctIndexer, press_speedIndexer, ink_pctIndexer, solvent_pctIndexer, esa_voltageIndexer, esa_amperageIndexer,
        //    waxIndexer, hardenerIndexer, roller_durometerIndexer, current_densityIndexer, anode_space_ratioIndexer, chrome_contentIndexer, lr});
        
        
        // Perceptron multicapa, aun no funciona
        //int layers[] = new int[]{35, 70, 35, 2};
        //MultilayerPerceptronClassifier mlp = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1).setMaxIter(100);
        //mlp.setFeaturesCol("featuresNormalized");
        //mlp.setLabelCol("band_typeIndex");
        
        //Normalizer normalizer = new Normalizer().setInputCol("features").setOutputCol("featuresNormalized").setP(1.0);
        
        //System.out.println("schema\n\n" + split[0].schema());
        //System.out.println("schema\n\n" + split[0].schema().json());
        
        //Pipeline pipelineMLP = new Pipeline().setStages(new PipelineStage[]{band_typeIndexer, assembler, normalizer, mlp});
        
        // Configurar un grid para buscar hiperparametros
        //ParamGridBuilder paramGridMLP = new ParamGridBuilder();
        //paramGridMLP.addGrid(mlp.stepSize(), new double[]{0.01, 0.001, 0.0015});
        
        //TrainValidationSplit trainValSplitMLP = new TrainValidationSplit().setEstimator(pipelineMLP).setEstimatorParamMaps(paramGridMLP.build())
        //        .setEvaluator(new BinaryClassificationEvaluator());
        
        //TrainValidationSplitModel modelMLP = trainValSplitMLP.fit(split[0]);
        //Dataset<Row> resultsMLP = modelMLP.transform(split[1]);
        //resultsMLP.show();
        
        //Calcular y mostrar las metricas de Accuracy y Confusion Matrix
        //MulticlassMetrics metrics = new MulticlassMetrics(resultsMLP.select("prediction","band_typeIndex"));
        
        //System.out.println("Test set accuracy: " + metrics.accuracy());
        //System.out.println("Confusion matrix: " + metrics.confusionMatrix());
        
        // c. Evalúe el resultado del entrenamiento de cada algoritmo mediante el conjunto de pruebas. Muestre su accuracy y matriz de confusión.
        // d. Salve en un fichero el modelo que mejor resultados arrojó.
        
        
    }
}
