package org.dma.gbdt4spark.data

import org.apache.spark.ml.linalg.Vector

case class Instance(label: Double, feature: Vector) {
  override def toString: String = {
    s"($label, $feature)"
  }
}
