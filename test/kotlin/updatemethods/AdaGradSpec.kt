/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package updatemethods

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class AdaGradSpec: Spek({

  describe("the AdaGrad update method") {

    on("get support structure") {

      val updateHelper = AdaGradMethod(learningRate = 0.001, epsilon = 1.0e-8)
      val updateableArray = Utils.buildUpdateableArray()

      it("should return a support structure of the expected type") {
        assertEquals(true, updateHelper.getSupportStructure(updateableArray) is AdaGradStructure)
      }
    }

    on("update") {

      val updateHelper = AdaGradMethod(learningRate = 0.001, epsilon = 1.0e-8)
      val updateableArray = Utils.buildUpdateableArray()
      val supportStructure = updateHelper.getSupportStructure(updateableArray) as AdaGradStructure

      supportStructure.secondOrderMoments.assignValues(Utils.supportArray2())

      updateHelper.update(array = updateableArray, errors = Utils.buildErrors())

      it("should match the expected updated array") {
        assertEquals(true, (updateableArray.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.39933, 0.39926, 0.49957, 0.999, 0.79978)),
          tolerance = 1.0e-5))
      }
    }
  }
})
