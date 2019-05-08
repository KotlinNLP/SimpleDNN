package core.layers.feedforward.squareddistance

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance.SquaredDistanceLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance.SquaredDistanceLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

object SquaredDistanceLayerUtils {

  /**
   *
   */
  fun buildLayer(): SquaredDistanceLayer = SquaredDistanceLayer(
      inputArray = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.5, -0.4))),
      params = SquaredDistanceLayerParameters(inputSize = 3, outputSize = 5),
      inputType =  LayerType.Input.Dense,
      id = 0)

  /**
   *
   */
  fun getParams53(): SquaredDistanceLayerParameters {

    val params = SquaredDistanceLayerParameters(inputSize = 5, outputSize = 3)

    params.B.values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.4, -0.5, 0.5, 0.5, -0.3),
            doubleArrayOf(0.6, 0.4, 0.4, 0.2, 0.4),
            doubleArrayOf(-0.5, 0.2, 0.1, -0.2, 0.4)
        )))

    return params
  }

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8))
}