/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.cfn.CFNLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.gru.GRULayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.lstm.LSTMLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters

/**
 *
 */
object LayerParametersFactory {

  operator fun invoke(inputSize: Int,
                      outputSize: Int,
                      connectionType: LayerType.Connection): LayerParameters =

    when(connectionType) {

      LayerType.Connection.Feedforward -> FeedforwardLayerParameters(
        inputSize = inputSize,
        outputSize = outputSize)

      LayerType.Connection.SimpleRecurrent -> SimpleRecurrentLayerParameters(
        inputSize = inputSize,
        outputSize = outputSize)

      LayerType.Connection.GRU -> GRULayerParameters(
        inputSize = inputSize,
        outputSize = outputSize)

      LayerType.Connection.LSTM -> LSTMLayerParameters(
        inputSize = inputSize,
        outputSize = outputSize)

      LayerType.Connection.CFN -> CFNLayerParameters(
        inputSize = inputSize,
        outputSize = outputSize)

      else -> throw RuntimeException("Invalid connection type: " + connectionType)
    }
}
