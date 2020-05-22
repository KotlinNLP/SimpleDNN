/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.tpr

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory


/**
 *
 */
internal object TPRLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(layersWindow: LayersWindow): TPRLayer<DenseNDArray> = TPRLayer(
      inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, 0.9, 0.1))),
      inputType = LayerType.Input.Dense,
      params = buildParams(),
      layersWindow = layersWindow,
      q = 0.001)

  /**
   *
   */
  fun buildParams(): TPRLayerParameters {

    val params = TPRLayerParameters(inputSize = 4, dRoles = 2, dSymbols = 3, nRoles = 3, nSymbols = 4)

    params.wInS.values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.2, 0.1, 0.3, -0.4),
            doubleArrayOf(0.3, -0.1, 0.9, 0.3),
            doubleArrayOf(0.4, 0.2, -0.3, 0.1),
            doubleArrayOf(0.6, 0.5, -0.4, 0.5)
        )))

    params.wInR.values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.3, 0.5, -0.5, -0.5),
            doubleArrayOf(0.5, 0.4, 0.1, 0.3),
            doubleArrayOf(0.6, 0.7, 0.8, 0.6)
        )))

    params.wRecS.values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.4, 0.2, -0.4, 0.5, 0.2, -0.5),
            doubleArrayOf(-0.2, 0.7, 0.8, -0.5, 0.5, 0.7),
            doubleArrayOf(0.4, -0.1, 0.1, 0.7, -0.1, 0.3),
            doubleArrayOf(0.3, 0.2, -0.7, -0.8, -0.3, 0.6)
        )))

    params.wRecR.values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.4, 0.8, -0.4, 0.7, 0.2, -0.5),
            doubleArrayOf(-0.2, 0.7, 0.8, -0.5, 0.3, 0.7),
            doubleArrayOf(0.3, -0.1, 0.1, 0.3, -0.1, 0.2)
        )))

    params.bS.values.assignValues(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.4, 0.8, 0.6))
    )

    params.bR.values.assignValues(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.2, -0.1))
    )

    params.s.values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.3, -0.2, -0.1, 0.5),
            doubleArrayOf(0.6, 0.7, 0.5, -0.6),
            doubleArrayOf(0.4, 0.2, 0.5, -0.6)
        )))

    params.r.values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.4, 0.3, 0.3),
            doubleArrayOf(0.3, 0.2, 0.1)
        )))

    return params
  }

  /**
   *
   */
  fun getOutputGold() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.57, 0.75, -0.15, 1.64, 0.45, 0.11))
}
