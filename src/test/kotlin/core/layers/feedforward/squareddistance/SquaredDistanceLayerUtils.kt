package core.layers.feedforward.squareddistance

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance.SquaredDistanceLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance.SquaredDistanceLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

object SquaredDistanceLayerUtils {

  /**
   *
   */
  fun buildLayer(): SquaredDistanceLayer<DenseNDArray> = SquaredDistanceLayer(
    inputArray = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.5, -0.4))),
    outputArray = AugmentedArray(1),
    params = this.getParams35(),
    inputType =  LayerType.Input.Dense,
    id = 0)

  /**
   *
   */
  fun getParams35(): SquaredDistanceLayerParameters {

    val params = SquaredDistanceLayerParameters(inputSize = 3, rank = 5)

    params.wB.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.4, 0.6, -0.5),
        doubleArrayOf(-0.5, 0.4, 0.2),
        doubleArrayOf(0.5, 0.4, 0.1),
        doubleArrayOf(0.5, 0.2, -0.2),
        doubleArrayOf(-0.3, 0.4, 0.4)
      )))

    return params
  }

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8))
}