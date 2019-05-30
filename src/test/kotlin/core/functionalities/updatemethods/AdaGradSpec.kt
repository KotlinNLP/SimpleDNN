/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class AdaGradSpec: Spek({

  describe("the AdaGrad update method") {

    context("update with dense errors") {

      context("update") {

        val updateHelper = AdaGradMethod(learningRate = 0.001, epsilon = 1.0e-8)
        val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
        val supportStructure = updateHelper.getSupportStructure(updatableArray)

        supportStructure.secondOrderMoments.assignValues(UpdateMethodsUtils.supportArray2())

        updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildDenseErrors())

        it("should match the expected updated array") {
          assertTrue {
            updatableArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.399331, 0.399258, 0.499569, 0.999, 0.799782)),
              tolerance = 1.0e-6)
          }
        }
      }
    }

    context("update with sparse errors") {

      context("update") {

        val updateHelper = AdaGradMethod(learningRate = 0.001, epsilon = 1.0e-8)
        val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
        val supportStructure = updateHelper.getSupportStructure(updatableArray)

        supportStructure.secondOrderMoments.assignValues(UpdateMethodsUtils.supportArray2())

        updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildSparseErrors())

        it("should match the expected updated array") {
          assertTrue {
            updatableArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.399258, 0.5, 1.0, 0.799443)),
              tolerance = 1.0e-6)
          }
        }
      }
    }
  }
})
