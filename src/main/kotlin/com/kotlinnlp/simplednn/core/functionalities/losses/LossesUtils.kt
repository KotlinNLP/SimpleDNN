/* Copyright 2016-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * Get the prediction errors respect to a gold one hot array using the Hinge Loss method.
 *
 * @param prediction a prediction array
 * @param goldIndex the index of the gold value
 * @param marginThreshold the max value of the
 *
 * @return the errors of the given prediction
 */
fun getErrorsByHingeLoss(prediction: DenseNDArray, goldIndex: Int, marginThreshold: Double = 1.0): DenseNDArray {

  val errors: DenseNDArray = DenseNDArrayFactory.zeros(prediction.shape)

  val highestScoringIncorrectIndex: Int = prediction.argMaxIndex(exceptIndex = goldIndex)

  val margin: Double = prediction[goldIndex] - prediction[highestScoringIncorrectIndex]

  if (margin < marginThreshold) {
    errors[goldIndex] = -1.0
    errors[highestScoringIncorrectIndex] = 1.0
  }

  return errors
}
