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
        Dataset<Row> df = spark.read().option("header", true).option("inferSchema", "true").csv("/home/maite/Documents/LogisticRegression/src/main/java/com/mycompany/logisticregression/bands.data");
        
        df.printSchema();  // inciso a
        df.show(10);       // inciso a
        System.out.println("Cantidad de filas del dataframe: " + df.count());
        
        // Eliminar las filas con valores nulos en el dataframe
        
        
        Dataset<Row> logregdataall = df.select(col("grain_screened"), 
                col("ink_color"), col("proof_ctd_ink"), col("blade_mfg"), 
                col("cylinder_division"), col("paper_type"), col("ink_type"),
                col("direct_steam"), col("solvent_type"), col("type_on_cylinder"),
                col("press_type"), col("press"), col("unit_number"),
                col("cylinder_size"), col("paper_mill_location"), col("plating_tank"),
                col("proof_cut"), col("viscosity"), col("caliper"), col("ink_temperature"),
                col("humifity"), col("roughness"), col("blade_pressure"), col("varnish_pct"),
                col("press_speed"), col("ink_pct"), col("solvent_pct"), col("esa_voltage"),
                col("esa_amperage"), col("wax"), col("hardener"), col("roller_durometer"),
                col("current_density"), col("anode_space_ratio"), col("chrome_content"),
                col("band_type"));
        
        logregdataall.printSchema();
        
        Dataset<Row> logregdata = logregdataall.na().drop(); // inciso c
        System.out.println("Cantidad de filas del dataframe despues de eliminar valores nulos: " + logregdata.count());
        
        //Mostrar los posibles valores que puede tomar el campo objetivo (target) que vamos a predecir, en este caso es band_type
        logregdata.groupBy("band_type").count().show();    // inciso b
        
        // Transformar los datos que tengas valores nominales, usar tecnicas de extraccion de caracteristicas
        
        
        StringIndexer grain_screenedIndexer = new StringIndexer().setInputCol("grain_screened").setOutputCol("grain_screenedIndex");
        StringIndexer ink_colorIndexer = new StringIndexer().setInputCol("ink_color").setOutputCol("ink_colorIndex");
        StringIndexer proof_ctd_inkIndexer = new StringIndexer().setInputCol("proof_ctd_ink").setOutputCol("proof_ctd_inkIndex");
        StringIndexer blade_mfgIndexer = new StringIndexer().setInputCol("blade_mfg").setOutputCol("blade_mfgIndex");
        StringIndexer cylinder_divisionIndexer = new StringIndexer().setInputCol("cylinder_division").setOutputCol("cylinder_divisionIndex");
        StringIndexer paper_typeIndexer = new StringIndexer().setInputCol("paper_type").setOutputCol("paper_typeIndex");
        StringIndexer ink_typeIndexer = new StringIndexer().setInputCol("ink_type").setOutputCol("ink_typeIndex");
        StringIndexer direct_steamIndexer = new StringIndexer().setInputCol("direct_steam").setOutputCol("direct_steamIndex");
        StringIndexer solvent_typeIndexer = new StringIndexer().setInputCol("solvent_type").setOutputCol("solvent_typeIndex");
        StringIndexer type_on_cylinderIndexer = new StringIndexer().setInputCol("type_on_cylinder").setOutputCol("type_on_cylinderIndex");
        StringIndexer press_typeIndexer = new StringIndexer().setInputCol("press_type").setOutputCol("press_typeIndex");
        StringIndexer pressIndexer = new StringIndexer().setInputCol("press").setOutputCol("pressIndex");
        StringIndexer unit_numberIndexer = new StringIndexer().setInputCol("unit_number").setOutputCol("unit_numberIndex");
        StringIndexer cylinder_sizeIndexer = new StringIndexer().setInputCol("cylinder_size").setOutputCol("cylinder_sizeIndex");
        StringIndexer paper_mill_locationIndexer = new StringIndexer().setInputCol("paper_mill_location").setOutputCol("paper_mill_locationIndex");
        StringIndexer plating_tankIndexer = new StringIndexer().setInputCol("plating_tank").setOutputCol("plating_tankIndex");
        StringIndexer band_typeIndexer = new StringIndexer().setInputCol("band_type").setOutputCol("band_typeIndex");
        
        OneHotEncoderEstimator encoder = new OneHotEncoderEstimator().setInputCols(new String[]{"grain_screenedIndex", "ink_colorIndex", "proof_ctd_inkIndex",
            "blade_mfgIndex", "cylinder_divisionIndex", "paper_typeIndex", "ink_typeIndex", "direct_steamIndex", "solvent_typeIndex", "type_on_cylinderIndex",
            "press_typeIndex", "pressIndex", "unit_numberIndex", "cylinder_sizeIndex", "paper_mill_locationIndex", "plating_tankIndex"}
            ).setOutputCols(new String[]{"grain_screenedVec", "ink_colorVec", "proof_ctd_inkVec", "blade_mfgVec", "cylinder_divisionVec", "paper_typeVec",
                "ink_typeVec", "direct_steamVec", "solvent_typeVec", "type_on_cylinderVec", "press_typeVec", "pressVec", "unit_numberVec", "cylinder_sizeVec",
                "paper_mill_locationVec", "plating_tankVec"});
       
        //Hacer 
        // 2. Seleccione al menos tres algoritmos de aprendizaje automático de acuerdo al problema identificado en el dataset y realice las siguientes acciones:
        // a. Cree un VectorAssembler a partir de los datos pre-procesados y divida de forma aleatoria el conjunto en dos partes un 70 % para entrenamiento y el 30 % para pruebas.
        VectorAssembler assembler = new VectorAssembler().setInputCols(new String[]{"grain_screened", "ink_color", "proof_ctd_ink",
            "blade_mfg", "cylinder_division", "paper_type", "ink_type", "direct_steam", "solvent_type", "type_on_cylinder",
            "press_type", "press", "unit_number", "cylinder_size", "paper_mill_location", "plating_tank",
            "proof_cut", "viscosity", "caliper", "ink_temperature", "humifity", "roughness", "blade_pressure", "varnish_pct", 
            "press_speed", "ink_pct", "solvent_pct", "esa_voltage", "esa_amperage", "wax", "hardener", "roller_durometer", 
            "current_density", "anode_space_ratio", "chrome_content"}).setOutputCol("features");
        
        Dataset<Row>[] split = logregdataall.randomSplit(new double[]{0.8, 0.2}, 1);
        split[0].show(10);
        
        // b. Entrene cada modelo elegido con dicha entrada y ajuste los hiper-parámetros correspondientes de forma automática.
        // c. Evalúe el resultado del entrenamiento de cada algoritmo mediante el conjunto de pruebas. Muestre su accuracy y matriz de confusión.
        // d. Salve en un fichero el modelo que mejor resultados arrojó.
        
        
    }
}
