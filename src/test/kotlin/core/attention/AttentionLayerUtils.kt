/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.attention

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism.AttentionMechanismLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object AttentionLayerUtils {

  /**
   *
   */
  fun buildAttentionParams(initializer: Initializer? = null) = AttentionMechanismLayerParameters(
    inputSize = 2,
    weightsInitializer = initializer
  ).apply {
    contextVector.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, -0.5)))
  }

  /**
   *
   */
  fun buildInputSequence(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.7, 0.9, 0.6)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, 0.7, -0.7, 0.8)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, -0.5, 0.0, 0.2))
  )

  /**
   *
   */
  fun buildAttentionSequence(inputSequence: List<DenseNDArray>): List<DenseNDArray> {

    val transformLayer: FeedforwardLayer<DenseNDArray> = buildTransformLayer()

    return inputSequence.map {
      transformLayer.setInput(it)
      transformLayer.forward()
      transformLayer.outputArray.values.copy()
    }
  }

  /**
   *
   */
  fun buildOutputErrors(): DenseNDArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.5, 0.1, -0.5))

  /**
   *
   */
  private fun buildTransformLayer(): FeedforwardLayer<DenseNDArray> = FeedforwardLayer(
    inputArray = AugmentedArray(size = 4),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray.zeros(2),
    params = buildTransformLayerParams(),
    activationFunction = Tanh,
    dropout = 0.0
  )

  /**
   *
   */
  fun buildTransformLayerParams() = FeedforwardLayerParameters(inputSize = 4, outputSize = 2).apply {

    unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(0.3, 0.4, 0.2, -0.2),
      doubleArrayOf(0.2, -0.1, 0.1, 0.6)
    )))

    unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, -0.4)))
  }
}
