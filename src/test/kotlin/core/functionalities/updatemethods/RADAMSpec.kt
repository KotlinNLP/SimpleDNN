/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class RADAMSpec : Spek({

  describe("the RADAM update method") {

    context("time step = 1") {

      context("update with dense errors") {

        context("update") {

          val updateHelper = RADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8)
          val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
          val supportStructure = updateHelper.getSupportStructure(updatableArray)

          supportStructure.firstOrderMoments.assignValues(UpdateMethodsUtils.supportArray1())
          supportStructure.secondOrderMoments.assignValues(UpdateMethodsUtils.supportArray2())

          updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildDenseErrors())

          it("should match the expected updated array") {
            assertTrue {
              updatableArray.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.399772, 0.399605, 0.499815, 0.995625, 0.799866)),
                tolerance = 1.0e-6)
            }
          }
        }
      }

      context("update with sparse errors") {

        context("update") {

          val updateHelper = RADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8)
          val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
          val supportStructure = updateHelper.getSupportStructure(updatableArray)

          supportStructure.firstOrderMoments.assignValues(UpdateMethodsUtils.supportArray1())
          supportStructure.secondOrderMoments.assignValues(UpdateMethodsUtils.supportArray2())

          updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildSparseErrors())

          it("should match the expected updated array") {
            assertTrue {
              updatableArray.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.399801, 0.399605, 0.49983, -269999.0, 0.799851)),
                tolerance = 1.0e-6)
            }
          }
        }
      }
    }

    context("time step = 6") {

      context("update with dense errors") {

        context("update") {

          val updateHelper = RADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8)
          val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
          val supportStructure = updateHelper.getSupportStructure(updatableArray)

          supportStructure.firstOrderMoments.assignValues(UpdateMethodsUtils.supportArray1())
          supportStructure.secondOrderMoments.assignValues(UpdateMethodsUtils.supportArray2())

          repeat(5) { updateHelper.newBatch() }

          updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildDenseErrors())

          it("should match the expected updated array") {
            assertTrue {
              updatableArray.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.399997, 0.399995, 0.499998, 0.999941, 0.799998)),
                tolerance = 1.0e-6)
            }
          }
        }
      }

      context("update with sparse errors") {

        context("update") {

          val updateHelper = RADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8)
          val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
          val supportStructure = updateHelper.getSupportStructure(updatableArray)

          supportStructure.firstOrderMoments.assignValues(UpdateMethodsUtils.supportArray1())
          supportStructure.secondOrderMoments.assignValues(UpdateMethodsUtils.supportArray2())

          repeat(5) { updateHelper.newBatch() }

          updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildSparseErrors())

          it("should match the expected updated array") {
            assertTrue {
              updatableArray.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.399997, 0.399995, 0.499998, -1486.902368, 0.799998)),
                tolerance = 1.0e-6)
            }
          }
        }
      }
    }
  }
})
