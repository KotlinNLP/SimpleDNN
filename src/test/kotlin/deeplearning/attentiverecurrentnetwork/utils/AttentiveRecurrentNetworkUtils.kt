/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.attentiverecurrentnetwork.utils

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.types.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.types.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.deeplearning.attention.attentiverecurrentnetwork.AttentiveRecurrentNetworkModel
import com.kotlinnlp.simplednn.deeplearning.attention.attentiverecurrentnetwork.AttentiveRecurrentNetworkParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object AttentiveRecurrentNetworkUtils {

  /**
   *
   */
  fun buildModel(): AttentiveRecurrentNetworkModel {

    val model = AttentiveRecurrentNetworkModel(
      inputSize = 2,
      attentionSize = 2,
      recurrentContextSize = 2,
      contextLabelSize = 2,
      outputSize = 2,
      contextActivation = Tanh(),
      contextRecurrenceType = LayerType.Connection.SimpleRecurrent,
      outputActivationFunction = Tanh())

    this.setTransformLayerParams(model)
    this.setAttentionNetworkParams(model)
    this.setRecurrentContextNetworkParams(model)
    this.setOutputNetworkParams(model)

    return model
  }

  /**
   *
   */
  fun buildInputSequence1(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.7)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, 0.7)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, -0.5))
  )

  /**
   *
   */
  fun buildInputSequence2(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.6, 0.4)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 0.3)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.1))
  )

  /**
   *
   */
  fun buildPredictionLabels(): List<DenseNDArray?> = listOf(
    null,
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.6, -0.6)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.8))
  )

  /**
   *
   */
  fun buildExpectedOutputs1(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.295800, -0.745375)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.346037, -0.796551)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.917978, -0.261493))
  )

  /**
   *
   */
  fun buildExpectedOutputs2(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.089257, -0.828523)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.468544, -0.791196)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.809604, -0.477566))
  )

  /**
   *
   */
  fun getExpectedInputErrors1(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.204547, -0.082545)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.209962, -0.082869)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.195195, -0.076396))
  )

  /**
   *
   */
  fun getExpectedLabelsErrors1(): List<DenseNDArray?> = listOf(
    null,
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.018696, 0.869881)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.018390, 0.014988))
  )

  /**
   *
   */
  fun getExpectedParamsErrors1(): AttentiveRecurrentNetworkParameters {

    val paramsErrors: AttentiveRecurrentNetworkParameters = this.buildModel().params.copy()

    this.setTransformLayerParamsErrors(paramsErrors)
    this.setAttentionNetworkParamsErrors(paramsErrors)
    this.setRecurrentContextNetworkParamsErrors(paramsErrors)
    this.setOutputNetworkParamsErrors(paramsErrors)

    return paramsErrors
  }

  /**
   *
   */
  fun getOutputErrors1(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.2, 0.9)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9, 0.0)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.1, 0.4))
  )

  /**
   *
   */
  private fun setTransformLayerParams(model: AttentiveRecurrentNetworkModel) {

    val params = model.transformParams

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(0.3, 0.4, 0.8, -0.6),
      doubleArrayOf(0.2, -0.1, -0.9, 1.0)
    )))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(
      doubleArrayOf(0.3, -0.4)
    ))
  }

  /**
   *
   */
  private fun setAttentionNetworkParams(model: AttentiveRecurrentNetworkModel) {

    val params = model.attentionParams.attentionParams

    params.contextVector.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, -0.5)))
  }

  /**
   *
   */
  private fun setRecurrentContextNetworkParams(model: AttentiveRecurrentNetworkModel) {

    val params = model.recurrentContextNetwork.model.paramsPerLayer[0] as SimpleRecurrentLayerParameters

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(0.7, -0.2, -0.4, 1.0),
      doubleArrayOf(0.0, 0.3, -0.7, -0.9)
    )))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, 0.6)))

    params.unit.recurrentWeights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(-0.9, 0.7),
      doubleArrayOf(0.2, -0.9)
    )))
  }

  /**
   *
   */
  private fun setOutputNetworkParams(model: AttentiveRecurrentNetworkModel) {

    val params = model.outputNetwork.model.paramsPerLayer[0] as FeedforwardLayerParameters

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(0.9, -0.7, -0.8, 0.7),
      doubleArrayOf(-0.6, -0.6, 0.8, 0.1)
    )))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(
      doubleArrayOf(0.1, -0.9)
    ))
  }

  /**
   *
   */
  private fun setTransformLayerParamsErrors(paramsErrors: AttentiveRecurrentNetworkParameters) {

    val errors = paramsErrors.transformParams

    errors.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(0.001396, -0.002158, 0.000219, -0.000696),
      doubleArrayOf(0.001896, -0.002861, 0.000625, -0.001793)
    )))

    errors.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(
      doubleArrayOf(-0.002094, -0.004802)
    ))
  }

  /**
   *
   */
  private fun setAttentionNetworkParamsErrors(paramsErrors: AttentiveRecurrentNetworkParameters) {

    val errors = paramsErrors.attentionParams.attentionParams

    errors.contextVector.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.004135, -0.003261)))
  }

  /**
   *
   */
  private fun setRecurrentContextNetworkParamsErrors(paramsErrors: AttentiveRecurrentNetworkParameters) {

    val errors = paramsErrors.recurrentContextParams.paramsPerLayer[0] as SimpleRecurrentLayerParameters

    errors.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(-0.063889, 0.095940, 0.182200, -0.166892),
      doubleArrayOf(0.031021, -0.046585, -0.090847, 0.097863)
    )))

    errors.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.307918, -0.149463)))

    errors.unit.recurrentWeights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(-0.003095, 0.008566),
      doubleArrayOf(-0.001419, 0.003926)
    )))
  }

  /**
   *
   */
  private fun setOutputNetworkParamsErrors(paramsErrors: AttentiveRecurrentNetworkParameters) {

    val errors = paramsErrors.outputParams.paramsPerLayer[0] as FeedforwardLayerParameters

    errors.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(0.043012, -0.064547, 0.059055, -0.173428),
      doubleArrayOf(-0.052429, 0.078677, 0.119006, -0.092441)
    )))

    errors.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(
      doubleArrayOf(-0.208488, 0.257541)
    ))
  }
}
