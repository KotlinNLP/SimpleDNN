/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.recurrent

import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ran.RANLayer
import com.kotlinnlp.simplednn.core.layers.StackedLayers
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The helper which calculates the importance scores of all the previous states of a given one, in a RAN neural network.
 */
internal class RANImportanceHelper {

  /**
   * The list of previous states structures.
   */
  private lateinit var prevStates: List<StackedLayers<*>>

  /**
   * The incremental product of the forget gates of the states, in reversed order.
   */
  private lateinit var incrementalForgetProd: DenseNDArray

  /**
   * The index of the RAN layer, among the layers of the network structure.
   */
  private var ranLayerIndex: Int = -1

  /**
   * Get the importance scores of the previous states respect of the last of a given sequence.
   * The scores values are in the range [0.0, 1.0].
   *
   * It is required that the network structures contain only a RAN layer.
   *
   * @param states the list of states seen by a recurrent neural processor
   *
   * @return the array containing the importance scores of the previous states
   */
  fun getImportanceScores(states: List<StackedLayers<*>>): DenseNDArray {

    this.initVars(states)

    val scores: DenseNDArray = DenseNDArrayFactory.emptyArray(Shape(this.prevStates.size))

    this.prevStates.indices.reversed().forEach { i ->

      val layer: RANLayer<*> = this.getRANLayer(i)

      scores[i] = layer.inputGate.values.prod(this.incrementalForgetProd).max()

      if (i > 0) this.incrementalForgetProd.assignProd(layer.forgetGate.values)
    }

    return scores
  }

  /**
   * Initialize the variables needed for the calculations.
   */
  private fun initVars(states: List<StackedLayers<*>>) {

    this.prevStates = states.subList(0, states.lastIndex)

    val lastStateLayers: List<Layer<*>> = states.last().layers
    this.ranLayerIndex = lastStateLayers.indexOfFirst { it is RANLayer }

    val lastStateLayer: RANLayer<*> = lastStateLayers[this.ranLayerIndex] as RANLayer
    this.incrementalForgetProd = lastStateLayer.forgetGate.values.copy()
  }

  /**
   * @param stateIndex the index of a state
   *
   * @return the RAN layer of the given state
   */
  private fun getRANLayer(stateIndex: Int): RANLayer<*> =
    this.prevStates[stateIndex].layers[this.ranLayerIndex] as RANLayer
}
