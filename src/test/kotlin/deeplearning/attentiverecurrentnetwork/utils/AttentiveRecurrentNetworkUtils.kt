/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.attentiverecurrentnetwork.utils

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.deeplearning.attentiverecurrentnetwork.AttentiveRecurrentNetworkModel
import com.kotlinnlp.simplednn.deeplearning.attentiverecurrentnetwork.AttentiveRecurrentNetworkParameters
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
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.431531, -0.360826)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.334841, -0.853749)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.907658, -0.252679))
  )

  /**
   *
   */
  fun buildExpectedOutputs2(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.058941, -0.538804)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.469632, -0.853854)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.786288, -0.468557))
  )

  /**
   *
   */
  fun getExpectedInputErrors1(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.310567, -0.164335)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.319349, -0.165902)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.293844, -0.153219))
  )

  /**
   *
   */
  fun getExpectedLabelsErrors1(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.464371, -0.636690)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.120753, 0.927892)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.020162, 0.012493))
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

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
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

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(0.7, -0.2, -0.4, 1.0),
      doubleArrayOf(0.0, 0.3, -0.7, -0.9)
    )))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, 0.6)))

    params.unit.recurrentWeights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(-0.9, 0.7),
      doubleArrayOf(0.2, -0.9)
    )))
  }

  /**
   *
   */
  private fun setOutputNetworkParams(model: AttentiveRecurrentNetworkModel) {

    val params = model.outputNetwork.model.paramsPerLayer[0] as FeedforwardLayerParameters

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
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

    errors.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(0.001854, -0.002859, -0.000462, -0.001199),
      doubleArrayOf(0.002297, -0.003461, -0.000493, -0.002329)
    )))

    errors.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(
      doubleArrayOf(-0.002562, -0.004909)
    ))
  }

  /**
   *
   */
  private fun setAttentionNetworkParamsErrors(paramsErrors: AttentiveRecurrentNetworkParameters) {

    val errors = paramsErrors.attentionParams.attentionParams

    errors.contextVector.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.005547, -0.004150)))
  }

  /**
   *
   */
  private fun setRecurrentContextNetworkParamsErrors(paramsErrors: AttentiveRecurrentNetworkParameters) {

    val errors = paramsErrors.recurrentContextParams.paramsPerLayer[0] as SimpleRecurrentLayerParameters

    errors.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(-0.037181, 0.055854, 0.105429, -0.095281),
      doubleArrayOf(0.031228, -0.046914, -0.090906, 0.096628)
    )))

    errors.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.169808, 0.076196)))

    errors.unit.recurrentWeights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(0.109232, 0.095000),
      doubleArrayOf(-0.104809, -0.081011)
    )))
  }

  /**
   *
   */
  private fun setOutputNetworkParamsErrors(paramsErrors: AttentiveRecurrentNetworkParameters) {

    val errors = paramsErrors.outputParams.paramsPerLayer[0] as FeedforwardLayerParameters

    errors.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(arrayOf(
      doubleArrayOf(0.045206, -0.067858, 0.147133, -0.082301),
      doubleArrayOf(-0.079239, 0.118969, 0.292940, 0.058410)
    )))

    errors.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(
      doubleArrayOf(-0.217984, 0.385762)
    ))
  }
}
