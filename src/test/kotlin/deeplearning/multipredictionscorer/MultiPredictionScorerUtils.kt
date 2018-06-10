/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.multipredictionscorer

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.types.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.utils.MultiMap
import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiPredictionModel
import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiPredictionNetworkConfig
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object MultiPredictionScorerUtils {

  /**
   *
   */
  fun buildModel(): MultiPredictionModel {

    val model = MultiPredictionModel(
      MultiPredictionNetworkConfig(
        inputSize = 2,
        inputType = LayerType.Input.Dense,
        hiddenSize = 2,
        hiddenActivation = Tanh(),
        outputSize = 2,
        outputActivation = null
      ),
      MultiPredictionNetworkConfig(
        inputSize = 2,
        inputType = LayerType.Input.Dense,
        hiddenSize = 2,
        hiddenActivation = Tanh(),
        outputSize = 2,
        outputActivation = null
      )
    )

    val params0In = model.networks[0].model.paramsPerLayer[0] as FeedforwardLayerParameters
    val params0Out = model.networks[0].model.paramsPerLayer[1] as FeedforwardLayerParameters
    val params1In = model.networks[1].model.paramsPerLayer[0] as FeedforwardLayerParameters
    val params1Out = model.networks[1].model.paramsPerLayer[1] as FeedforwardLayerParameters

    params0In.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(0.3, -0.9),
      doubleArrayOf(-0.5, -0.5)
    )))

    params0In.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(
      doubleArrayOf(-0.3, 0.7)
    ))

    params0Out.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(-0.5, -0.1),
      doubleArrayOf(0.6, 0.5)
    )))

    params0Out.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(
      doubleArrayOf(0.1, 0.5)
    ))

    params1In.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(-0.5, -0.8),
      doubleArrayOf(-0.8, 0.7)
    )))

    params1In.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(
      doubleArrayOf(-0.5, 0.0)
    ))

    params1Out.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(-0.1, -0.5),
      doubleArrayOf(-0.6, 0.4)
    )))

    params1Out.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(
      doubleArrayOf(0.1, -1.0)
    ))

    return model
  }

  /**
   *
   */
  fun buildInputFeaturesMap(): MultiMap<DenseNDArray> = MultiMap(mapOf(
    Pair(
      0,
      mapOf(
        Pair(0, DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, 0.2))),
        Pair(1, DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 0.3))),
        Pair(2, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.6, -0.9)))
      )
    ),
    Pair(
      1,
      mapOf(
        Pair(0, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.8))),
        Pair(1, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, -1.0)))
      )
    )
  ))

  /**
   *
   */
  fun buildOutputErrors(): MultiMap<DenseNDArray> = MultiMap(mapOf(
    Pair(
      0,
      mapOf(
        Pair(0, DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, 0.3))),
        Pair(1, DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, -0.9))),
        Pair(2, DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.1)))
      )
    ),
    Pair(
      1,
      mapOf(
        Pair(0, DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.1))),
        Pair(1, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.2)))
      )
    )
  ))
}
