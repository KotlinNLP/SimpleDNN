/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.newrecirculation

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.utils.ItemsPool

/**
 * New Recirculation network.
 *
 * @property model the network model
 * @param recallThreshold the threshold of mean absolute error (between the real and the imaginary inputs) beyond
 *                        which the recirculation process is triggered (default = 0.001)
 * @param trainingLearningRate the learning rate of the training (default = 0.01)
 */
class NewRecirculationNetwork(
  val model: NewRecirculationModel,
  private val recallThreshold: Double = 0.001,
  private val trainingLearningRate: Double = 0.01,
  override val id: Int = 0
) : ItemsPool.IDItem {

  companion object {

    /**
     * The max number of recall iterations.
     */
    private const val MAX_RECALL_ITERATIONS = 1.0e04
  }

  /**
   *
   */
  private val realInput = AugmentedArray(values = DenseNDArrayFactory.zeros(Shape(this.model.inputSize)))

  /**
   *
   */
  private val realOutput = AugmentedArray(values = DenseNDArrayFactory.zeros(Shape(this.model.hiddenSize)))

  /**
   *
   */
  private val imaginaryInput = AugmentedArray(values = DenseNDArrayFactory.zeros(Shape(this.model.inputSize)))

  /**
   *
   */
  private val imaginaryOutput = AugmentedArray(values = DenseNDArrayFactory.zeros(Shape(this.model.hiddenSize)))

  /**
   *
   */
  private val paramsErrors: FeedforwardLayerParameters = this.model.params.copy()

  /**
   *
   */
  val meanAbsError: Double get() = this.realInput.values.sub(this.imaginaryInput.values).abs().sum()

  /**
   * Set activation functions.
   */
  init {

    if (this.model.activationFunction != null) {
      this.realOutput.setActivation(this.model.activationFunction)
      this.imaginaryOutput.setActivation(this.model.activationFunction)
    }
  }

  /**
   * Reconstruct a given array.
   *
   * @param inputArray the input array
   * @param trainingMode whether the parameters must be trained during the re-entry
   *
   * @return the [inputArray] reconstruction
   */
  fun reconstruct(inputArray: DenseNDArray, trainingMode: Boolean = false): DenseNDArray {
    require(inputArray.length == this.model.inputSize)

    this.realInput.assignValues(inputArray)

    this.calcImaginaryInput()

    this.reEntry(trainingMode = trainingMode)

    return this.imaginaryInput.values
  }

  /**
   * Calculate the imaginary input.
   * The [realInput] must be already set.
   *
   *  yR = f(w (dot) xR + b)
   *  xI = r * xR + (1 - r) * w' (dot) yR
   */
  private fun calcImaginaryInput() {

    val r: Double = this.model.lambda
    val w: DenseNDArray = this.model.params.unit.weights.values as DenseNDArray
    val b: DenseNDArray = this.model.params.unit.biases.values as DenseNDArray
    val xR: DenseNDArray = this.realInput.values
    val yR: DenseNDArray = this.realOutput.values
    val xI: DenseNDArray = this.imaginaryInput.values

    yR.assignDot(w, xR).assignSum(b)
    this.realOutput.activate()

    // Note of optimization: double transposition of two 1-dim arrays instead of a bigger 2-dim one
    xI.assignSum(xR.prod(r), yR.t.dot(w).t.assignProd(1 - r))
  }

  /**
   * Perform the re-entry technique for the recall process.
   */
  private fun reEntry(trainingMode: Boolean = false) {

    var iterations = 0

    while (this.meanAbsError >= this.recallThreshold && iterations++ < MAX_RECALL_ITERATIONS) {

      if (trainingMode) this.backward()

      this.realInput.assignValues(this.imaginaryInput.values)

      this.calcImaginaryInput()
    }
  }

  /**
   * Perform the backward calculating errors end updating the parameters.
   *
   *  yI = r * yR + (1 - r) * f(w (dot) xI)
   */
  private fun backward() {

    val r: Double = this.model.lambda
    val w: DenseNDArray = this.model.params.unit.weights.values as DenseNDArray
    val b: DenseNDArray = this.model.params.unit.biases.values as DenseNDArray
    val yR: DenseNDArray = this.realOutput.values
    val xI: DenseNDArray = this.imaginaryInput.values
    val yI: DenseNDArray = this.imaginaryOutput.values

    yI.assignDot(w, xI).assignSum(b)
    this.imaginaryOutput.activate()

    yI.assignProd(1 - r).assignSum(yR.prod(r))

    this.calcParamsErrors()
    this.update()
  }

  /**
   * Calculate the parameters errors, assign them and update the parameters.
   *
   *  gb = yI - yR
   *  gw = (yI - yR) (dot) xI + ((xI - xR) (dot) yR)'
   */
  private fun calcParamsErrors() {

    val xR: DenseNDArray = this.realInput.values
    val yR: DenseNDArray = this.realOutput.values
    val xI: DenseNDArray = this.imaginaryInput.values
    val yI: DenseNDArray = this.imaginaryOutput.values

    val gw: DenseNDArray = this.paramsErrors.unit.weights.values as DenseNDArray
    val gb: DenseNDArray = this.paramsErrors.unit.biases.values as DenseNDArray

    val gx: DenseNDArray = xI.sub(xR)
    val gy: DenseNDArray = yI.sub(yR)

    gw.assignDot(gy, xI.t).assignSum(yR.dot(gx.t))
    gb.assignValues(gy)
  }

  /**
   * Update the model parameters.
   */
  private fun update() {

    val lr: Double = this.trainingLearningRate

    val w: DenseNDArray = this.model.params.unit.weights.values as DenseNDArray
    val b: DenseNDArray = this.model.params.unit.biases.values as DenseNDArray

    val gw: DenseNDArray = this.paramsErrors.unit.weights.values as DenseNDArray
    val gb: DenseNDArray = this.paramsErrors.unit.biases.values as DenseNDArray

    w.assignSub(gw.assignProd(lr))
    b.assignSub(gb.assignProd(lr))
  }
}
