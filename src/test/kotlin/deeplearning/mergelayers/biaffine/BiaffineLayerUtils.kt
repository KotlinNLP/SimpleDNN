package deeplearning.mergelayers.biaffine

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.deeplearning.mergelayers.biaffine.BiaffineLayerParameters
import com.kotlinnlp.simplednn.deeplearning.mergelayers.biaffine.BiaffineLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object BiaffineLayerUtils {

  /**
   *
   */
  fun buildLayer(): BiaffineLayerStructure<DenseNDArray> = BiaffineLayerStructure(
    inputArray1 = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9))),
    inputArray2 = AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, -0.2, 0.6))),
    outputArray = AugmentedArray(size = 2),
    params = buildParams(),
    activationFunction = Tanh()
  )

  /**
   *
   */
  fun buildParams(): BiaffineLayerParameters {

    val params = BiaffineLayerParameters(inputSize1 = 2, inputSize2 = 3, outputSize = 2)

    params.w1.values.assignValues(
      DenseNDArrayFactory.arrayOf(arrayOf(
        doubleArrayOf(0.3, 0.8),
        doubleArrayOf(0.8, -0.7)
      )))

    params.w2.values.assignValues(
      DenseNDArrayFactory.arrayOf(arrayOf(
        doubleArrayOf(0.6, 0.5, -0.9),
        doubleArrayOf(0.3, -0.3, 0.3)
      )))

    params.b.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, -0.4)))

    params.w[0].values.assignValues(
      DenseNDArrayFactory.arrayOf(arrayOf(
        doubleArrayOf(-0.4, 0.2),
        doubleArrayOf(0.2, 0.4),
        doubleArrayOf(0.0, 0.5)
      )))

    params.w[1].values.assignValues(
      DenseNDArrayFactory.arrayOf(arrayOf(
        doubleArrayOf(-0.2, 0.9),
        doubleArrayOf(0.5, 0.0),
        doubleArrayOf(-0.1, -0.1)
      )))

    return params
  }

  /**
   *
   */
  fun getOutputGold() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, -0.3))
}
