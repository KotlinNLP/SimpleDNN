/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.feedforward

import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkStructure
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.NDArray

/**
 *
 * @param neuralNetwork neuralNetwork
 */
class FeedforwardNeuralProcessor(override val neuralNetwork: NeuralNetwork) : NeuralProcessor {

  /**
   *
   */
  var structure = NetworkStructure(
    layersConfiguration = this.neuralNetwork.layersConfiguration,
    params = this.neuralNetwork.model)

  /**
   * The errors of the network model parameters
   */
  private val backwardParamsErrors: NetworkParameters = this.neuralNetwork.parametersFactory()

  /**
   *
   * @return
   */
  override fun getOutput(copy: Boolean): NDArray {
    return if (copy) {
      this.structure.outputLayer.outputArray.values.copy()
    } else {
      this.structure.outputLayer.outputArray.values
    }
  }

  /**
   *
   */
  override fun getParamsErrors(): NetworkParameters {
    val paramsError = this.neuralNetwork.parametersFactory()
    paramsError.assignValues(this.backwardParamsErrors)
    return paramsError
  }

  /**
   *
   * @return
   */
  fun getInputErrors(copy: Boolean = true): NDArray {
    return if (copy) {
      this.structure.inputLayer.inputArray.errors.copy()
    } else {
      this.structure.inputLayer.inputArray.errors
    }
  }

  /**
   *
   * @param featuresArray features
   */
  fun forward(featuresArray: NDArray, useDropout: Boolean = false): NDArray {

    this.structure.forward(featuresArray, useDropout = useDropout)

    return this.structure.outputLayer.outputArray.values
  }

  /**
   *
   * @param outputErrors the errors on the output of the network
   * @param propagateToInput propagateErrorsToInput
   * @return the avgLoss respect to the output of the network
   */
  fun backward(outputErrors: NDArray,
               propagateToInput: Boolean = false) {
    this.structure.backward(
      outputErrors = outputErrors,
      paramsErrors = this.backwardParamsErrors,
      propagateToInput = propagateToInput)
  }

}
