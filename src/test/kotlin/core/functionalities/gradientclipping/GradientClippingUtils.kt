package core.functionalities.gradientclipping

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import core.optimizer.ParamsErrorsAccumulatorUtils

object GradientClippingUtils {

  /**
   *
   */
  fun buildParams(): List<ParamsArray> {

    val params = FeedforwardLayerParameters(inputSize = 4, outputSize = 5)

    params.unit.weights.values.assignValues(buildDenseParams3())

    params.unit.biases.values.assignValues(buildDenseParams1())

    return params.paramsList.map { ParamsArray(it.values) }
  }

  /**
   *
   */
  fun buildErrors(): ParamsErrorsList {

    val accumulator = ParamsErrorsAccumulator()
    val params = buildParams()

    val gw1 = params[0].buildDenseErrors(buildWeightsErrorsValues1())
    val gb1 = params[1].buildDenseErrors(buildBiasesErrorsValues1())
    accumulator.accumulate(listOf(gw1, gb1))

    return accumulator.getParamsErrors()
  }

  /**
   *
   */
  fun buildDenseParams1() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0))

  /**
   *
   */
  fun buildDenseParams3() = DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(0.0, 0.0, 0.0, 0.0),
      doubleArrayOf(0.0, 0.0, 0.0, 0.0),
      doubleArrayOf(0.0, 0.0, 0.0, 0.0),
      doubleArrayOf(0.0, 0.0, 0.0, 0.0),
      doubleArrayOf(0.0, 0.0, 0.0, 0.0)
  ))

  /**
   *
   */
  fun buildBiasesErrorsValues1() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.9, 0.7, 0.4, 0.8, 0.1))

  /**
   *
   */
  fun buildWeightsErrorsValues1() = DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(0.5, 0.6, -0.8, -0.6),
      doubleArrayOf(0.7, -0.4, 0.1, -0.8),
      doubleArrayOf(0.7, -0.7, 0.3, 0.5),
      doubleArrayOf(0.8, -0.9, 0.0, -0.1),
      doubleArrayOf(0.4, 1.0, -0.7, 0.8)
  ))

}