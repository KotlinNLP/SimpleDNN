/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralnetwork

import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import java.io.Serializable

/**
 * [NetworkParameters] contains all the parameters of the layers defined in [layersConfiguration],
 * grouped per layer.
 *
 * @param layersConfiguration a list of configurations, one per layer
 */
class NetworkParameters(val layersConfiguration: List<LayerConfiguration>) :
  Serializable,
  Iterable<UpdatableArray<*>> {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The iterator inner class which iterates over all the parameters of all the layers
   */
  private inner class LayerParametersIterator: Iterator<UpdatableArray<*>> {
    /**
     *
     */
    private var curLayerIndex: Int = 0

    /**
     *
     */
    private var layerParamsIterator: Iterator<UpdatableArray<*>> = this@NetworkParameters.paramsPerLayer.first().iterator()

    /**
     *
     */
    override fun hasNext(): Boolean {
      return this.curLayerIndex < this@NetworkParameters.paramsPerLayer.lastIndex || this.layerParamsIterator.hasNext()
    }

    /**
     *
     */
    override fun next(): UpdatableArray<*> {

      if (!this.layerParamsIterator.hasNext()) {
        this.layerParamsIterator = this@NetworkParameters.paramsPerLayer[++this.curLayerIndex].iterator()
      }

      return this.layerParamsIterator.next()
    }
  }

  /**
   * An [Array] containing a [LayerParameters] for each layer.
   *
   * In [layersConfiguration] layers are defined as a list [x, y, z], but the structure
   * contains layers as input-output pairs [x-y, y-z].
   * The output of a layer is a reference of the input of the next layer.
   */
  val paramsPerLayer: Array<LayerParameters> = Array(size = layersConfiguration.size - 1, init = {
    LayerParametersFactory(
      inputSize = layersConfiguration[it].size,
      outputSize = layersConfiguration[it + 1].size,
      connectionType = layersConfiguration[it + 1].connectionType!!)
  })

  /**
   * The iterator to use to iterate over all the parameters of all the layers
   *
   * @return the iterator of all the parameters
   */
  override fun iterator(): Iterator<UpdatableArray<*>> = this.LayerParametersIterator()

  /**
   * Assign the values of each parameter of [assigningParameters] to the parameters of this [NetworkParameters].
   *
   * @param assigningParameters the [NetworkParameters] to assign to this
   */
  fun assignValues(assigningParameters: NetworkParameters) {
    this.zip(assigningParameters).forEach {
      it.first.values.assignValues(it.second.values)
    }
  }

  /**
   * Sum the values each parameter of [addingParameters] to the parameters of this [NetworkParameters].
   *
   * @param addingParameters the [NetworkParameters] to add to this
   */
  fun assignSum(addingParameters: NetworkParameters) {
    this.zip(addingParameters).forEach {
      it.first.values.assignSum(it.second.values)
    }
  }

  /**
   * Divide the values of each parameter by [n].
   *
   * @param n an integer number
   */
  fun assignDiv(n: Int) {
    if (n > 1) {
      val nDouble = n.toDouble()
      this.forEach { it.values.assignDiv(nDouble) }
    }
  }

  /**
   * Assign 0.0 to all parameters
   */
  fun reset() {
    this.forEach { it.values.assignValues(0.0) }
  }

  /**
   * Initialize the parameters with a random generator and fixed biases values.
   *
   * @param randomGenerator randomGenerator
   * @param biasesInitValue biasesInitValue
   */
  fun initialize(randomGenerator: RandomGenerator, biasesInitValue: Double) {
    this.paramsPerLayer.forEach {
      it.initialize(randomGenerator = randomGenerator, biasesInitValue = biasesInitValue)
    }
  }
}
