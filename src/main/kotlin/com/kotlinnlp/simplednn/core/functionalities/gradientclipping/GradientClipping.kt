/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.gradientclipping

import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList

class GradientClipping {


  /**
   * Clip [paramsErrors] in-place at specified [clipValue]
   */
  fun clipByValue (paramsErrors: ParamsErrorsList, clipValue: Double) {

    paramsErrors.map {
      for (i in 0 until it.values.rows)
        for (j in 0 until it.values.columns)
          if (it.values[i, j].toDouble() < -clipValue || it.values[i, j].toDouble() > clipValue)
            it.values[i, j] = clipValue
    }

  }
}