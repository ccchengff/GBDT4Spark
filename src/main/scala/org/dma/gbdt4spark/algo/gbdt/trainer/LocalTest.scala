package org.dma.gbdt4spark.algo.gbdt.trainer

import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkConf, SparkContext}
import org.dma.gbdt4spark.algo.gbdt.predictor.FPGBDTPredictor
import org.dma.gbdt4spark.common.Global.Conf._
import org.dma.gbdt4spark.tree.param.GBDTParam
import org.dma.gbdt4spark.util.Maths

object LocalTest {

  def main(args: Array[String]): Unit = {
    @transient val conf = new SparkConf().setMaster("local").setAppName("test")

    val trainPath = "data/agaricus/agaricus_127d_train.libsvm"
    conf.set(ML_TRAIN_DATA_PATH, trainPath)
    val validPath = "data/agaricus/agaricus_127d_train.libsvm"
    conf.set(ML_VALID_DATA_PATH, validPath)
    val modelPath = "tmp/model"
    conf.set(ML_MODEL_PATH, modelPath)
    val predPath = "tmp/pred"
    conf.set(ML_PREDICT_DATA_PATH, predPath)

    conf.set(ML_NUM_CLASS, "2")
    conf.set(ML_NUM_FEATURE, "149")
    conf.set(ML_NUM_WORKER, "1")
    conf.set(ML_LOSS_FUNCTION, "binary:logistic")
    conf.set(ML_EVAL_METRIC, "error,auc")
    conf.set(ML_LEARN_RATE, "0.1")
    conf.set(ML_GBDT_MAX_DEPTH, "3")

    @transient implicit val sc = SparkContext.getOrCreate(conf)

    val param = new GBDTParam
    param.numClass = conf.getInt(ML_NUM_CLASS, DEFAULT_ML_NUM_CLASS)
    param.numFeature = conf.get(ML_NUM_FEATURE).toInt
    param.featSampleRatio = conf.getDouble(ML_FEATURE_SAMPLE_RATIO, DEFAULT_ML_FEATURE_SAMPLE_RATIO).toFloat
    param.numWorker = conf.get(ML_NUM_WORKER).toInt
    param.numThread = conf.getInt(ML_NUM_THREAD, DEFAULT_ML_NUM_THREAD)
    param.lossFunc = conf.get(ML_LOSS_FUNCTION)
    param.evalMetrics = conf.get(ML_EVAL_METRIC, DEFAULT_ML_EVAL_METRIC).split(",").map(_.trim).filter(_.nonEmpty)
    param.learningRate = conf.getDouble(ML_LEARN_RATE, DEFAULT_ML_LEARN_RATE).toFloat
    param.histSubtraction = conf.getBoolean(ML_GBDT_HIST_SUBTRACTION, DEFAULT_ML_GBDT_HIST_SUBTRACTION)
    param.lighterChildFirst = conf.getBoolean(ML_GBDT_LIGHTER_CHILD_FIRST, DEFAULT_ML_GBDT_LIGHTER_CHILD_FIRST)
    param.fullHessian = conf.getBoolean(ML_GBDT_FULL_HESSIAN, DEFAULT_ML_GBDT_FULL_HESSIAN)
    param.numSplit = conf.getInt(ML_GBDT_SPLIT_NUM, DEFAULT_ML_GBDT_SPLIT_NUM)
    param.numTree = conf.getInt(ML_GBDT_TREE_NUM, DEFAULT_ML_GBDT_TREE_NUM)
    param.maxDepth = conf.getInt(ML_GBDT_MAX_DEPTH, DEFAULT_ML_GBDT_MAX_DEPTH)
    val maxNodeNum = Maths.pow(2, param.maxDepth + 1) - 1
    param.maxNodeNum = conf.getInt(ML_GBDT_MAX_NODE_NUM, maxNodeNum) min maxNodeNum
    param.minChildWeight = conf.getDouble(ML_GBDT_MIN_CHILD_WEIGHT, DEFAULT_ML_GBDT_MIN_CHILD_WEIGHT).toFloat
    param.minNodeInstance = conf.getInt(ML_GBDT_MIN_NODE_INSTANCE, DEFAULT_ML_GBDT_MIN_NODE_INSTANCE)
    param.minSplitGain = conf.getDouble(ML_GBDT_MIN_SPLIT_GAIN, DEFAULT_ML_GBDT_MIN_SPLIT_GAIN).toFloat
    param.regAlpha = conf.getDouble(ML_GBDT_REG_ALPHA, DEFAULT_ML_GBDT_REG_ALPHA).toFloat
    param.regLambda = conf.getDouble(ML_GBDT_REG_LAMBDA, DEFAULT_ML_GBDT_REG_LAMBDA).toFloat max 1.0f
    param.maxLeafWeight = conf.getDouble(ML_GBDT_MAX_LEAF_WEIGHT, DEFAULT_ML_GBDT_MAX_LEAF_WEIGHT).toFloat
    println(s"Hyper-parameters:\n$param")

    try {
      val trainer = new SparkFPGBDTTrainer(param)
      trainer.initialize(trainPath, validPath)
      val model = trainer.train()
      trainer.save(model, modelPath)

      val predictor = new FPGBDTPredictor
      predictor.loadModel(sc, modelPath)
      predictor.predict(sc, validPath, predPath)
    } catch {
      case e: Exception =>
        e.printStackTrace()
    } finally {
      Array(modelPath, predPath).foreach { dir =>
        val path = new Path(dir)
        val fs = path.getFileSystem(sc.hadoopConfiguration)
        if (fs.exists(path)) fs.delete(path, true)
      }
    }
  }

}
