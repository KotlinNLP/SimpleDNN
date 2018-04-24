package com.kotlinnlp.simplednn.core.layers.recurrent.indrnn

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.layers.ParametersUnit
import com.kotlinnlp.simplednn.core.layers.RecurrentLayerUnit

/**
 * The parameters associated to a [RecurrentLayerUnit].
 *
 * @property inputSize input size
 * @property outputSize output size
 * @param sparseInput whether the weights connected to the input are sparse or not (default false)
 * @param meProp whether to use the 'meProp' errors propagation algorithm (params are sparse) (default false)
 */
class IndRNNParametersUnit(
  inputSize: Int,
  outputSize: Int,
  sparseInput: Boolean = false,
  meProp: Boolean = false
) : ParametersUnit(
  inputSize = inputSize,
  outputSize = outputSize,
  sparseInput = sparseInput,
  meProp = meProp) {

  /**
   *
   */
  val recurrentWeights: UpdatableArray<*> = this.buildUpdatableArray(
    dim1 = this.outputSize,
    sparse = meProp)
}
