/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.mergelayers.biaffine

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.layers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.*
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray

/**
 * The helper which executes the backward on a biaffine [layer].
 *
 * @property layer the [BiaffineLayerStructure] in which the backward is executed
 */
class BiaffineBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: BiaffineLayerStructure<InputNDArrayType>
) : BackwardHelper<InputNDArrayType> {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun backward(paramsErrors: LayerParameters<*>, propagateToInput: Boolean) {

    this.layer.applyOutputActivationDeriv()

    val gwx1: Array<DenseNDArray> = this.getWX1ArraysGradients()

    this.assignParamsGradients(paramsErrors = paramsErrors as BiaffineLayerParameters, wx1Errors = gwx1)

    if (propagateToInput) {
      this.assignLayerGradients(wxErrors = gwx1)
    }
  }

  /**
   *
   */
  private fun getWX1ArraysGradients(): Array<DenseNDArray> {
    // TODO: actually the wx errors are Sparse if the input is SparseBinary: calculations should be optimized

    val x2: InputNDArrayType = this.layer.inputArray2.values
    val gy: DenseNDArray = this.layer.outputArray.errors

    return Array(
      size = this.layer.params.outputSize,
      init = { i ->
        val gwxi: Double = gy[i]

        when (x2) {
          is DenseNDArray -> x2.prod(gwxi)
          is SparseBinaryNDArray -> {
            // TODO: actually the wx arrays are sparse, replace with the commented code to optimize further calculations
            val tmpArray = DenseNDArrayFactory.zeros(shape = x2.shape)
            val mask: NDArrayMask = x2.mask

            for (k in 0 until mask.dim1.size) {
              tmpArray[mask.dim1[k], mask.dim2[k]] = gwxi
            }

            tmpArray

//            val zeros = DenseNDArrayFactory.zeros(shape = x2.shape)
//            val mask: NDArrayMask = x2.mask
//            SparseNDArrayFactory.arrayOf(
//              activeIndicesValues = Array(
//                size = mask.dim1.size,
//                init = { k -> SparseEntry(Indices(mask.dim1[k], mask.dim2[k]), gwi) }
//              ),
//              shape = x2.shape
//            )
          }
          else -> throw RuntimeException("Invalid input type")
        }
      }
    )
  }

  /**
   *
   */
  private fun assignParamsGradients(paramsErrors: BiaffineLayerParameters, wx1Errors: Array<DenseNDArray>) {
    // TODO: actually the wx errors are Sparse if the input is SparseBinary: calculations should be optimized

    val x1: InputNDArrayType = this.layer.inputArray1.values
    val x1T : InputNDArrayType = x1.T
    val x2: InputNDArrayType = this.layer.inputArray2.values

    val gy: DenseNDArray = this.layer.outputArray.errors
    val gwArrays: Array<UpdatableArray<*>> = paramsErrors.w
    val gw1: NDArray<*> = paramsErrors.w1.values
    val gw2: NDArray<*> = paramsErrors.w2.values
    val gb: NDArray<*> = paramsErrors.b.values

    gwArrays.forEachIndexed { i, gwArray ->
      val gwi: DenseNDArray = gwArray.values as DenseNDArray
      val gwx1i: DenseNDArray = wx1Errors[i]
      gwi.assignDot(gwx1i, x1T)
    }

    gw1.assignDot(gy, x1T)
    gw2.assignDot(gy, x2.T)
    gb.assignValues(gy)
  }

  /**
   *
   */
  private fun assignLayerGradients(wxErrors: Array<DenseNDArray>) {
    // TODO: actually the wx errors are Sparse if the input is SparseBinary: calculations should be optimized

    val w1: DenseNDArray = this.layer.params.w1.values as DenseNDArray
    val w2: DenseNDArray = this.layer.params.w2.values as DenseNDArray
    val wArrays: Array<UpdatableArray<*>> = this.layer.params.w

    val gy: DenseNDArray = this.layer.outputArray.errors
    val gyT: DenseNDArray = gy.T

    this.layer.inputArray1.assignErrorsByDotT(gyT, w1)
    this.layer.inputArray2.assignErrorsByDotT(gyT, w2)

    wxErrors.forEachIndexed { i, wxError ->
      val wi: DenseNDArray = wArrays[i].values as DenseNDArray
      val wx1i: DenseNDArray = this.layer.wx1Arrays[i]

      this.layer.inputArray1.errors.assignSum(wxError.T.dot(wi))
      this.layer.inputArray2.errors.assignSum(wx1i.prod(gy[i]))
    }
  }
}
