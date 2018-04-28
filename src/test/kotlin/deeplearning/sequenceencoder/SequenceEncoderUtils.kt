/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.sequenceencoder

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.encoders.sequenceencoder.SequenceFeedforwardNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object SequenceEncoderUtils {

  /**
   *
   */
  fun buildInputSequence(): Array<DenseNDArray> = arrayOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.3, -0.8)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, -0.9, 0.6)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, 0.3, -0.6))
  )

  /**
   *
   */
  fun buildOutputErrorsSequence(): Array<DenseNDArray> = arrayOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.2)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, 0.0)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, -0.9))
  )

  /**
   *
   */
  fun buildNetwork(): SequenceFeedforwardNetwork {

    val network = SequenceFeedforwardNetwork(
      inputType = LayerType.Input.Dense,
      inputSize = 3,
      outputSize = 2,
      outputActivation = Tanh())

    initParameters(network.network.model.paramsPerLayer[0] as FeedforwardLayerParameters)

    return network
  }


  /**
   *
   */
  private fun initParameters(params: FeedforwardLayerParameters) {

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(-0.7, 0.3, -1.0),
      doubleArrayOf(0.8, -0.6, 0.4)
    )))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.2, -0.9)))
  }
}
