package org.dma.gbdt4spark.algo.gbdt.predictor

import org.apache.spark.{SparkConf, SparkContext, SparkEnv}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.dma.gbdt4spark.algo.gbdt.tree.{GBTNode, GBTTree}
import org.dma.gbdt4spark.common.Global.Conf._
import org.dma.gbdt4spark.objective.ObjectiveFactory
import org.dma.gbdt4spark.util.DataLoader

object FPGBDTPredictor {

  def predict(predictor: FPGBDTPredictor, instances: RDD[Vector]): RDD[Array[Float]] = {
    val bcPredictor = instances.sparkContext.broadcast(predictor)
    instances.map(bcPredictor.value.predict)
  }

  def main(args: Array[String]): Unit = {
    @transient val conf = new SparkConf()
    @transient implicit val sc = SparkContext.getOrCreate(conf)

    val modelPath = conf.get(ML_MODEL_PATH)
    val validPath = conf.get(ML_VALID_DATA_PATH)

    println(s"Reading model from $modelPath")
    val model = sc.objectFile[Seq[GBTTree]](modelPath).collect().head
    val predictor = new FPGBDTPredictor(model)

    println(s"Predicting dataset $validPath")
    val instances = DataLoader.loadLibsvmDP(validPath, model.head.getParam.numFeature).cache()
    val labels = instances.map(_.label.toFloat).collect()
    val preds = predict(predictor, instances.map(_.feature)).collect().flatten
    instances.unpersist()

    println(s"Evaluating predictions")
    val evalMetrics = ObjectiveFactory.getEvalMetrics(
      conf.get(ML_EVAL_METRIC, DEFAULT_ML_EVAL_METRIC)
        .split(",").map(_.trim).filter(_.nonEmpty)
    )
    println(evalMetrics.map(evalMetric => {
      val kind = evalMetric.getKind
      val metric = evalMetric.eval(preds, labels)
      s"$kind[$metric]"
    }).mkString(", "))
  }
}

class FPGBDTPredictor(forest: Seq[GBTTree]) extends Serializable {

  def predict(instance: Vector): Array[Float] = {
    val param = forest.head.getParam
    val preds = Array.ofDim[Float](if (param.numClass == 2) 1 else param.numClass)
    forest.foreach(tree => {
      var node = tree.getRoot
      while (!node.isLeaf) {
        if (node.getSplitEntry.flowTo(instance) == 0)
          node = node.getLeftChild.asInstanceOf[GBTNode]
        else
          node = node.getRightChild.asInstanceOf[GBTNode]
      }
      if (param.numClass == 2) {
        preds(0) += node.getWeight * param.learningRate
      } else {
        val weights = node.getWeights
        for (k <- 0 until param.numClass)
          preds(k) += weights(k)
      }
    })
    preds
  }

}
